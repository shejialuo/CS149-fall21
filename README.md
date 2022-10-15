# CS149

This is my code and writing assignment for studying [CS149](https://gfxcourses.stanford.edu/cs149/fall21/lecture/).

However, for the video, you can watch the CMU 15-418, which is just
like CS149:

+ [Bilibili](https://www.bilibili.com/video/BV16k4y1z7z9)
+ [Panopto](https://scs.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx#folderID=%22f62c2297-de88-4e63-aff2-06641fa25e98%22)

However, I find a better [video](http://www.cs.cmu.edu/afs/cs/academic/class/15418-f18/www/schedule.html) which is taught by Randal E. Bryant.

## Environment setup

For assignment 1, we need to install ISPC. It is easy to install in ArchLinux.

```sh
sudo pacman -S ispc
```

For assignment 2, we need nothing.

For assignment 3, you could install cuda in ArchLinux. However, I pay money to
rent a GPU VPS due to the reason that I have no NVIDIA GPU.

For assignment 4, you could install `openmp` package in ArchLinux.

```sh
sudo pacman -S openmp
```

For extra assignment, you still need ispc and also you need to use Intel math
library to compare the efficiency if you like.

```sh
sudo pacman -S intel-oneapi-mkl
```

---

At this time, except the extra assignment. I have already finished all the parts
of the matrix multiplication. Well, in this whole series, I have learned a lot.
Thanks CS149 team for all the efforts!
