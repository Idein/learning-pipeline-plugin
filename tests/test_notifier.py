import learning_pipeline_plugin


def test_notifier(capfd):
    notifier = learning_pipeline_plugin.notifier.Notifier()
    message = "test message"
    notifier.notify(message)

    out, _ = capfd.readouterr()
    assert message in out
