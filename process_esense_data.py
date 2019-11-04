import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('seaborn-whitegrid')
data_file = "RAW_2.txt" # Name/Path of Raw Data File

data = np.zeros([1, 7])

with open(data_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    start = 0
    for row in csv_reader:
        if not row:
            continue
        timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = (float(row[0]), float(row[1]), float(row[2]), # Unpack Row
                                                                  float(row[3]), float(row[4]), float(row[5]), float(row[6]))

        sample_time = datetime.datetime.fromtimestamp(timestamp / 1000.0)#.strftime('%c') # Convert Timestamp to Local Time
        if (line_count == 0):
            start = sample_time

        offset_time = (sample_time - start).total_seconds() # Seconds since first sample

        data = np.append(data, np.array([[offset_time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]]), axis=0)
        line_count += 1


# Plot Accelerometer/Gyroscope Data
x = data[:, 0]

plt.figure(0)
plt.subplot(3, 1, 1)

plt.plot(x, data[:, 1], '.-')
plt.title('Accelerometer: X')

plt.subplot(3, 1, 2)
plt.plot(x, data[:, 2], '.-')
plt.title('Accelerometer: Y')

plt.subplot(3, 1, 3)
plt.plot(x, data[:, 3], '.-')
plt.title('Accelerometer: Z')
plt.xlabel('Seconds (s)')

plt.figure(1)
plt.subplot(3, 1, 1)

plt.plot(x, data[:, 4], '.-')
plt.title('Gyroscope: X')

plt.subplot(3, 1, 2)
plt.plot(x, data[:, 5], '.-')
plt.title('Gyroscope: Y')

plt.subplot(3, 1, 3)
plt.plot(x, data[:, 6], '.-')
plt.title('Gyroscope: Z')
plt.xlabel('Seconds (s)')

plt.show()


