obj-m += driver.o
driver-objs += recallmem-driver.o utils.o

all:
	make -C ../../build MODULE_SRC=$(shell pwd) linux-modules MODULE_OUTPUT=$(shell pwd)/../out/kernel_modules
