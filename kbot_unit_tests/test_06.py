import asyncio
from pykos import KOS
from utils.telemetry import log_telemetry, TelemetryLogger, plot_latest_logs

async def main():
    async with KOS(ip="10.33.11.164") as kos:
        telemetry_logger = TelemetryLogger(kos, [11, 12])
        await telemetry_logger.start()
        await asyncio.sleep(10.0)
        await telemetry_logger.stop()
        plot_latest_logs()
    # or you can use it as:
    # async with KOS(ip="10.33.11.164") as kos:
    #     await log_telemetry(
    #         kos,
    #         actuator_ids=[11, 12, 13],
    #         duration=10.0,
    #     )

asyncio.run(main())