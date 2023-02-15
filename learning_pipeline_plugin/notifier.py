from datetime import datetime, timezone

import actfw_core


class AbstractNotifier:
    """
    Abstract class to notify information to actcast
    """
    def notify(self, message: str) -> None:
        """Notify input 'message'.
        User should implement this method for notifying with a special
        set of keys.
        (e.g. in order to not trigger a previously used (Actcast's) Cast rule)
        """
        raise NotImplementedError


class NullNotifier(AbstractNotifier):
    def notify(self, message: str) -> None:
        pass


class Notifier(AbstractNotifier):
    """
    Simple notifier object.
    """
    def notify(self, message: str) -> None:
        actfw_core.notify(
            [
                {
                    "info": message,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]
        )
