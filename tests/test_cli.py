"""Tests for the command-line entry point.

The JSON mode is a contract the TypeScript wrapper depends on, so it is tested
as a contract: exactly one JSON object on stdout, and nothing else.
"""

from __future__ import annotations

import json

import pytest

from agentic_loop import cli

from .fakes import FakeAnthropic, text_message, tool_use_message


@pytest.fixture
def fake_client(monkeypatch):
    """Replace client construction so no API key or network is needed."""

    def install(responses):
        client = FakeAnthropic(responses)
        monkeypatch.setattr(cli, "_build_client", lambda: client)
        return client

    return install


class TestPrintTools:
    def test_prints_the_catalogue_without_calling_the_api(self, capsys) -> None:
        assert cli.main(["--print-tools"]) == cli.EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert [tool["name"] for tool in payload["tools"]] == [
            "calculator",
            "word_stats",
            "document_search",
        ]
        assert all(tool["strict"] is True for tool in payload["tools"])


class TestJsonMode:
    def test_stdout_is_exactly_one_json_object(self, fake_client, capsys) -> None:
        fake_client([text_message("Rome")])
        exit_code = cli.main(["--json", "--no-stream", "What is the capital of Italy?"])

        captured = capsys.readouterr()
        payload = json.loads(captured.out)  # would raise if anything else leaked to stdout
        assert exit_code == cli.EXIT_OK
        assert payload["ok"] is True
        assert payload["result"]["final_text"] == "Rome"
        assert payload["result"]["stop_reason"] == "completed"

    def test_includes_the_trace_and_tool_calls(self, fake_client, capsys) -> None:
        fake_client(
            [
                tool_use_message(("c1", "calculator", {"expression": "6 * 7"})),
                text_message("42"),
            ]
        )
        cli.main(["--json", "--no-stream", "What is 6 * 7?"])

        payload = json.loads(capsys.readouterr().out)
        assert payload["result"]["tool_invocations"][0]["name"] == "calculator"
        assert "run_start" in [event["kind"] for event in payload["trace"]]

    def test_streamed_text_goes_to_stderr_so_stdout_stays_parsable(
        self, fake_client, capsys
    ) -> None:
        fake_client([text_message("streamed words")])
        cli.main(["--json", "hello"])

        captured = capsys.readouterr()
        json.loads(captured.out)
        assert "streamed" in captured.err

    def test_an_incomplete_run_exits_nonzero(self, fake_client, capsys) -> None:
        fake_client([tool_use_message(("c1", "calculator", {"expression": "1 + 1"}))])
        exit_code = cli.main(["--json", "--no-stream", "--max-iterations", "1", "loop"])

        payload = json.loads(capsys.readouterr().out)
        assert exit_code == cli.EXIT_INCOMPLETE
        assert payload["ok"] is False
        assert payload["result"]["stop_reason"] == "budget_exhausted"

    def test_a_configuration_error_is_reported_as_json(self, capsys) -> None:
        exit_code = cli.main(["--json", "--task-budget", "10", "hello"])
        payload = json.loads(capsys.readouterr().out)
        assert exit_code == cli.EXIT_ERROR
        assert payload["ok"] is False
        assert payload["error"]["type"] == "ConfigurationError"

    def test_a_client_failure_is_reported_as_json(self, monkeypatch, capsys) -> None:
        from agentic_loop.errors import AgenticLoopError

        def boom():
            raise AgenticLoopError("no credentials")

        monkeypatch.setattr(cli, "_build_client", boom)
        exit_code = cli.main(["--json", "hello"])
        payload = json.loads(capsys.readouterr().out)
        assert exit_code == cli.EXIT_ERROR
        assert payload["error"]["message"] == "no credentials"


class TestHumanMode:
    def test_writes_the_answer_to_stdout_and_the_summary_to_stderr(
        self, fake_client, capsys
    ) -> None:
        fake_client([text_message("Rome")])
        assert cli.main(["--no-stream", "capital of Italy?"]) == cli.EXIT_OK

        captured = capsys.readouterr()
        assert "Rome" in captured.out
        assert "[completed]" in captured.err

    def test_reports_the_reason_a_run_stopped_early(self, fake_client, capsys) -> None:
        fake_client([tool_use_message(("c1", "calculator", {"expression": "1 + 1"}))])
        cli.main(["--no-stream", "--max-iterations", "1", "loop"])

        captured = capsys.readouterr()
        assert "[budget_exhausted]" in captured.err
        assert "iteration cap" in captured.err


class TestArgumentHandling:
    def test_passes_budget_flags_through(self, fake_client, capsys) -> None:
        client = fake_client([text_message()])
        cli.main(
            [
                "--json",
                "--no-stream",
                "--model",
                "claude-sonnet-5",
                "--effort",
                "low",
                "--max-tokens",
                "2048",
                "--system",
                "Be terse.",
                "hello",
            ]
        )
        capsys.readouterr()
        request = client.last_request
        assert request["model"] == "claude-sonnet-5"
        assert request["output_config"]["effort"] == "low"
        assert request["max_tokens"] == 2048
        assert request["system"] == "Be terse."

    def test_reads_the_prompt_from_stdin(self, fake_client, capsys, monkeypatch) -> None:
        import io

        client = fake_client([text_message("piped")])
        monkeypatch.setattr("sys.stdin", io.StringIO("prompt from stdin"))
        cli.main(["--json", "--no-stream"])
        capsys.readouterr()
        assert client.last_request["messages"][0]["content"] == "prompt from stdin"

    def test_reports_a_missing_prompt_in_the_same_shape(self, monkeypatch, capsys) -> None:
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        exit_code = cli.main(["--json"])

        payload = json.loads(capsys.readouterr().out)
        assert exit_code == cli.EXIT_ERROR
        assert payload["error"]["type"] == "ConfigurationError"
        assert "no prompt" in payload["error"]["message"]


class TestApiErrorTranslation:
    """The SDK's exceptions must become reported errors, never tracebacks."""

    def test_maps_a_missing_credential_typeerror(self) -> None:
        # The SDK validates credentials on the first request and raises a bare
        # TypeError, not a typed authentication error.
        exc = TypeError(
            "Could not resolve authentication method. Expected one of api_key, "
            "auth_token, or credentials to be set."
        )
        translated = cli._translate_api_error(exc)
        assert translated.kind == "authentication"
        assert "ANTHROPIC_API_KEY" in str(translated)

    @pytest.mark.parametrize(
        ("class_name", "expected_kind"),
        [
            ("AuthenticationError", "authentication"),
            ("PermissionDeniedError", "permission"),
            ("RateLimitError", "rate_limit"),
            ("NotFoundError", "not_found"),
            ("BadRequestError", "bad_request"),
            ("APIConnectionError", "connection"),
            ("APITimeoutError", "connection"),
        ],
    )
    def test_maps_typed_sdk_errors_by_identity(self, class_name: str, expected_kind: str) -> None:
        exc = type(class_name, (Exception,), {})("boom")
        assert cli._translate_api_error(exc).kind == expected_kind

    def test_classifies_by_status_code_when_the_type_is_unknown(self) -> None:
        exc = type("SomeNewError", (Exception,), {"status_code": 503})("upstream down")
        translated = cli._translate_api_error(exc)
        assert translated.kind == "server_error"
        assert "503" in str(translated)

        exc = type("SomeNewError", (Exception,), {"status_code": 409})("conflict")
        assert cli._translate_api_error(exc).kind == "api_error"

    def test_falls_back_without_losing_the_original_type(self) -> None:
        translated = cli._translate_api_error(RuntimeError("something odd"))
        assert translated.kind == "unexpected"
        assert "RuntimeError" in str(translated)

    def test_reports_an_sdk_failure_instead_of_raising(self, monkeypatch, capsys) -> None:
        class ExplodingClient:
            @property
            def messages(self):
                raise TypeError("Could not resolve authentication method.")

        monkeypatch.setattr(cli, "_build_client", ExplodingClient)
        exit_code = cli.main(["--json", "--no-stream", "hello"])

        payload = json.loads(capsys.readouterr().out)
        assert exit_code == cli.EXIT_ERROR
        assert payload["error"]["type"] == "ApiError"
        assert "authentication failed" in payload["error"]["message"]

    def test_debug_flag_re_raises_the_original_exception(self, monkeypatch) -> None:
        class ExplodingClient:
            @property
            def messages(self):
                raise RuntimeError("original failure")

        monkeypatch.setattr(cli, "_build_client", ExplodingClient)
        with pytest.raises(RuntimeError, match="original failure"):
            cli.main(["--json", "--no-stream", "--debug", "hello"])
