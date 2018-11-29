#!/bin/env bash

aws s3 sync --delete ../docs/build/html/ s3://docs.pewresearch.tech/pewanalytics/
