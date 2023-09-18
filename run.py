import os
import sys
import time


def gpu_info(gpu_index=0):
    info = os.popen("nvidia-smi|grep %").read().split("\n")[gpu_index].split("|")
    power = int(info[1].split()[-3][:-1])
    used_memory = int(info[2].split("/")[0].strip()[:-3])
    all_memory = int(info[2].split("/")[1].strip()[:-3])
    free_memory = all_memory - used_memory
    return power, free_memory


def run(interval=2):
    cmd = "python main.py"  # 当资源空闲时执行的程序
    gpu_power, free_memory = gpu_info()
    i = 0
    while free_memory < 19000:  # set waiting condition
        gpu_power, free_memory = gpu_info()
        i = i % 5
        symbol = "monitoring: " + ">" * i + " " * (10 - i - 1) + "|"
        gpu = "gpu: 0 |"
        gpu_power_str = "gpu power:%d W |" % gpu_power
        gpu_memory_str = "gpu free_memory:%d MiB |" % free_memory
        sys.stdout.write(
            "\r" + gpu + gpu_memory_str + " " + gpu_power_str + " " + symbol
        )
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print("\n" + cmd)

    os.system(cmd)


if __name__ == "__main__":
    run()
