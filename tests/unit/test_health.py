import asyncio

from httpx import ASGITransport, AsyncClient

from clearframe.main import app


def test_health_endpoint() -> None:
    async def request() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "version": "0.1.0"}

    asyncio.run(request())

