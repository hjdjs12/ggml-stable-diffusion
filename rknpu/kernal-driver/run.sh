mkfifo /tmp/fifo-mm-uk
mkfifo /tmp/fifo-mm-ku
chmod 0666 /tmp/fifo-mm-uk
chmod 0666 /tmp/fifo-mm-ku
mkfifo /tmp/fifo-ta-uk-0
mkfifo /tmp/fifo-ta-ku-0
chmod 0666 /tmp/fifo-ta-uk-0
chmod 0666 /tmp/fifo-ta-ku-0
mkfifo /tmp/fifo-ta-uk-1
mkfifo /tmp/fifo-ta-ku-1
chmod 0666 /tmp/fifo-ta-uk-1
chmod 0666 /tmp/fifo-ta-ku-1

insmod /mnt/host/nw-el1/driver.ko

cp /mnt/host/out/*.ta /lib/optee_armtz
/mnt/host/out/mm-daemon &
/mnt/host/out/ta-daemon 0 &
/mnt/host/out/ta-daemon 1 &