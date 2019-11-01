import asyncio
from bleak import discover, BleakClient

devices = []

# Populate Bluetooth Devices List
def scan_devices():

    async def run():
        global devices
        device_routine = await discover()
        for d in device_routine:
            print ("\t" + str(d))
            devices.append((d.address, d.name))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())


def connect_to_device(address, name):
    print ("\n Attempting to Connect to {} at {}...".format(name, address))
    async def run(address, loop):
        async with BleakClient(address, loop=loop) as client:
            model_number = await client.read_gatt_char(MODEL_NBR_UUID)
            print("Model Number: {0}".format("".join(map(chr, model_number))))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(address, loop))


if __name__ == "__main__":

    # Scan BT Devices and Find eSense Device by Name
    name, address = None, None
    while address is None:
        print ("Scanning BT Devices...")
        scan_devices()
        for device in devices:
            if "eSense" in device[1]:
                name = device[1]
                address = device[0]
    print ("Found {} at Address {}".format(name, address))

    # Connect to eSense Device
    connect_to_device(address, name)
    
            
