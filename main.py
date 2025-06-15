import asyncio
from data_processing import read_matrix_from_csv, generate_commands
from bluetooth_communication import BluetoothController
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

class RegionController:
    def __init__(self, zone_id, rgb_flags, total_duration, ble_controller):
        self.zone_id = int(zone_id)  # Ensure zone_id is an integer
        self.rgb_flags = [int(x) for x in rgb_flags]  # Convert to integers
        self.total_duration = total_duration
        self.ble_controller = ble_controller
        self.subtasks = self._build_subtasks()
        self.step_duration = total_duration / len(self.subtasks) if self.subtasks else 0

    def _build_subtasks(self):
        tasks = []
        if self.rgb_flags[0] == 1:  # Explicitly check for 1
            tasks.append([self.zone_id, 1, 0, 0])
        if self.rgb_flags[1] == 1:
            tasks.append([self.zone_id, 0, 1, 0])
        if self.rgb_flags[2] == 1:
            tasks.append([self.zone_id, 0, 0, 1])
        return tasks

    async def run(self):
        if not self.subtasks:
            log(f"[ZONE {self.zone_id}] No light therapy tasks to execute")
            return

        log(f"[ZONE {self.zone_id}] rgb_flags: {self.rgb_flags}, subtasks: {self.subtasks}")
        for matrix in self.subtasks:
            zone_id, r, g, b = matrix
            cmd = generate_commands([matrix])[0]
            await self.ble_controller.send_command(cmd)
            log(f"[ZONE {self.zone_id}] Emitting light [{r}, {g}, {b}], duration {self.step_duration:.2f}s")
            await asyncio.sleep(0.1)  # Brief delay to ensure command is sent
            await asyncio.sleep(self.step_duration)

            # Turn off the current zone's light
            shutdown_cmd = f"PIN{self.zone_id + 6}:0,0,0"
            await self.ble_controller.send_command(shutdown_cmd)
            await asyncio.sleep(0.1)

async def control_all_regions_parallel(matrix, total_duration):
    ble_controller = BluetoothController()
    await ble_controller.connect()

    if not ble_controller.client:
        return

    log(f"[INFO] Starting parallel light therapy, total duration: {total_duration} seconds")

    # Create a controller for each zone
    tasks = []
    for row in matrix:
        zone, r, g, b = row
        log(f"[DEBUG] Zone: {zone}, rgb_flags: [{r}, {g}, {b}]")
        controller = RegionController(zone, [r, g, b], total_duration, ble_controller)
        tasks.append(controller.run())

    await asyncio.gather(*tasks)

    # Turn off all zones
    shutdown_matrix = [[row[0], 0, 0, 0] for row in matrix]
    shutdown_cmds = generate_commands(shutdown_matrix)
    for cmd in shutdown_cmds:
        await ble_controller.send_command(cmd)
        log(f"Shutdown command: {cmd}")

    await asyncio.sleep(1)  # Give Arduino time to respond
    log("[INFO] All light therapy completed and turned off")

    await ble_controller.disconnect()

def main():
    file_path = "input_matrix.csv"
    input_matrix = read_matrix_from_csv(file_path)

    if not input_matrix:
        print("[ERROR] Input matrix is empty or CSV file format is invalid")
        return

    total_time = 24  #Total time
    asyncio.run(control_all_regions_parallel(input_matrix, total_time))

if __name__ == "__main__":
    main()
