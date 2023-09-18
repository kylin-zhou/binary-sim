import os
import sys
import time

import schedule

def job():
    run_time_cmd = "ps -eo pid,comm,cmd,start,etime | grep 'python main.py' | grep -v grep | sort -k 5 | head -n 1"
    run_time = os.popen(run_time_cmd).read().strip().split(" ")[-1]
    sys.stdout.write("\r" + f"run time : {run_time}")
    sys.stdout.flush()
    
    run_hour = run_time.split(":")[0]
    if run_time and int(run_hour) >= 60:
        # kill
        print("kill job")
        kill_cmd = "kill -9 `ps -ef|grep 'python main.py'|grep -v 'grep'|awk '{print $2}'`"
        os.system(kill_cmd)
        watchdog = False

watchdog = True

schedule.every(10).minutes.do(job) # 每隔 10 分运行一次 job 函数

while True and watchdog:
    schedule.run_pending()   # 运行所有可以运行的任务
    time.sleep(360)