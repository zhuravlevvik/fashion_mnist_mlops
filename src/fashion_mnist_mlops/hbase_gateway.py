"""Authenticated HTTP boundary in front of an internal HBase Thrift endpoint."""

import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from fashion_mnist_mlops.events import PredictionEvent
from fashion_mnist_mlops.hbase_repository import HBaseRepository
from fashion_mnist_mlops.secret_provider import get_hbase_gateway_credentials

CredentialsProvider = Callable[[], tuple[str, str]]

# Saved as legacy (lab 2)
# def environment_credentials() -> tuple[str, str]:
#     return os.environ["HBASE_GATEWAY_USERNAME"], os.environ["HBASE_GATEWAY_PASSWORD"]


def configured_credentials() -> tuple[str, str]:
    credentials = get_hbase_gateway_credentials()
    return credentials.username, credentials.password


def create_gateway_app(
    repository: HBaseRepository | None = None,
    credentials_provider: CredentialsProvider = configured_credentials,
) -> FastAPI:
    repository = repository or HBaseRepository.from_env()
    security = HTTPBasic()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        repository.ensure_schema()
        yield

    app = FastAPI(
        title="Fashion-MNIST HBase gateway",
        version="0.2.0",
        lifespan=lifespan,
    )

    def authenticate(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
        expected_username, expected_password = credentials_provider()

        username_ok = secrets.compare_digest(credentials.username, expected_username)
        password_ok = secrets.compare_digest(credentials.password, expected_password)

        if not (username_ok or password_ok):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid HBase gateway credentials",
                headers={"WWW-Authenticate": "Basic"}
            )
        return credentials.username

    @app.get("/health")
    def health() -> dict[str, str]:
        repository.ping()
        return {"status": "ok", "backend": "hbase"}

    @app.post("/predictions", status_code=status.HTTP_201_CREATED)
    def store_prediction(
        event: PredictionEvent,
        _username: str = Depends(authenticate)
    ) -> dict[str, str]:
        repository.put(event)
        return {"prediction_id": event.prediction_id, "status": "stored"}

    @app.get("/predictions/{prediction_id}")
    def get_prediction(
        prediction_id: str,
        _username: str = Depends(authenticate)
    ) -> PredictionEvent:
        event = repository.get(prediction_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return event

    return app


app = create_gateway_app()


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
