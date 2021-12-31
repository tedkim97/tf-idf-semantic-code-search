# Configure VM on fresh bootup

sudo apt update
sudo apt install python3-pip libjpeg-dev zlib1g-dev unzip
pip3 install numpy scipy matplotlib gensim pandas jupyter sklearn


sudo lsblk

sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
sudo mkdir -p /mnt/disks/attached
sudo mount -o discard,defaults /dev/sdb /mnt/disks/attached
sudo chmod a+w /mnt/disks/attached

sudo cp /etc/fstab /etc/fstab.backup
sudo blkid /dev/sdb 