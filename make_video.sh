#!/bin/bash

ffmpeg -framerate 15 -i frames/%04d.png -c:v libx264 -r 15 -pix_fmt yuv420p -b:v 20M out.mp4