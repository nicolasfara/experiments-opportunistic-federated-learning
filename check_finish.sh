#!/bin/bash
for file in data/*
  do LASTLINE="$(tail -n 1 "$file")"
  if [ "${LASTLINE:0:1}" != "#" ]
    then echo "${file}"
  fi
done