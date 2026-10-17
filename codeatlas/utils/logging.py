import logging

from codeatlas.observability.timing import watch_provider_retries


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    # The provider client sleeps through its own 429 backoffs inside the model call, where no
    # stage can see them. It says how long it is about to sleep; this listens for that.
    watch_provider_retries()
