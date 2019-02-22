# cuBLAS

## Dynamic
```
$ nvcc matrix_multiple.cu -lcublas -o test_shared
```

## Static
```
$ nvcc matrix_multiple.cu -lcublas_static -lculibos -o test_static
```

## Check Dependencies and Size
```
dzi@dorian-ubun:~/deeplearning/MYGITHUB/cuda/cublas_apps$ ldd test_static 
	linux-vdso.so.1 =>  (0x00007ffec33ac000)
	librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007f4571bc7000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f45719aa000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f45717a6000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f4571423000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f4571059000)
	/lib64/ld-linux-x86-64.so.2 (0x0000563482dee000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f4570d50000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f4570b39000)
dzi@dorian-ubun:~/deeplearning/MYGITHUB/cuda/cublas_apps$ du -h test_static 
10M	test_static


dzi@dorian-ubun:~/deeplearning/MYGITHUB/cuda/cublas_apps$ ldd test_shared 
	linux-vdso.so.1 =>  (0x00007ffcdd995000)
	libcublas.so.9.0 => /usr/local/cuda/lib64/libcublas.so.9.0 (0x00007fac40ed2000)
	librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007fac40cae000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fac40a91000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fac4088c000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fac4050a000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fac40140000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fac3fe36000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fac3fc20000)
	/lib64/ld-linux-x86-64.so.2 (0x0000559fe6e14000)
dzi@dorian-ubun:~/deeplearning/MYGITHUB/cuda/cublas_apps$ du -h test_shared 
604K	test_shared
```

## Note
To run it with shared library, make sure LD_LIBRARY_PATH contains the path of libcublas
```
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64
$ ./test_shared

OR

$ LD_LIBRARY_PATH=/usr/local/cuda/lib64 ./test_shared
```




