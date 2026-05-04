#!/bin/bash
#Created by Mrugank Dake for batch Anatomical segmentation
#Date:11/04/2023
#Updated for TMS_ori project: 05/04/2026

# Define the array of subject numbers
subs=(1 2 3 5 6 7 8 9 11 12 14 16 17 18 19 20 21 22 23 24 25 26 28)

original_dir=$(pwd)

# Define the maximum number of cores to use
MAX_CORE=55

# Get the length of the subjects array
NUM_SUBS=${#subs[@]}

# Check if the number of subjects exceeds the maximum number of cores available
if [ $NUM_SUBS -gt $MAX_CORE ]; then
    echo "Not enough cores! Reduce # of subjects or number of cores/subject!"
    exit 1
fi

DATE=$(date +%Y%m%d_%H%M%S)

# Initialize the core number counter
CORENUM=0

# Loop over the subject numbers
for sub in "${subs[@]}"; do
    sub_id=$(printf "sub%02d" $sub)
    # Paths for TMS_ori project
    input_dir="/d/DATD/datd/TMS_ori/brainsight/${sub_id}"
    work_dir="/d/DATD/datd/TMS_ori/SIMNIBS_output/${sub_id}"
    LOG_DIR="/d/DATD/datd/TMS_ori/SIMNIBS_output/logs"
    LOG_FILE="${LOG_DIR}/charm_${sub_id}_${DATE}.log"
    
    mkdir -p "$LOG_DIR"
    mkdir -p "$work_dir"

    echo "Starting processing for Subject: $sub_id on Core: $CORENUM (Log: $LOG_FILE)"
    
    (
      # Copy T1.nii from brainsight if not in work_dir
      if [ ! -e "${work_dir}/T1.nii" ]; then
        if [ -e "${input_dir}/T1.nii" ]; then
          cp "${input_dir}/T1.nii" "${work_dir}/"
        else
          echo "Error: T1.nii not found in $input_dir. Skipping $sub_id."
          exit 1
        fi
      fi

      # Optionally copy T2.nii if it exists in input_dir
      if [ ! -e "${work_dir}/T2.nii" ] && [ -e "${input_dir}/T2.nii" ]; then
        cp "${input_dir}/T2.nii" "${work_dir}/"
      fi

      cd "$work_dir"
      
      # Check if T2.nii exists
      if [ -e "T2.nii" ]; then
        command=(charm "$sub_id" "T1.nii" "T2.nii" "--forceqform")
      else
        command=(charm "$sub_id" "T1.nii" "--forceqform")
      fi
      
      {
        echo "Standard Output:"
        "${command[@]}"
        echo "Standard Error:" >&2
      } &> "$LOG_FILE" &
    ) &

    # Note: taskset is Linux-specific. If running on Mac, this will fail.
    if command -v taskset &> /dev/null; then
        taskset --cpu-list $CORENUM -p $! &> /dev/null
    fi

    ((CORENUM++))
    if [ $CORENUM -ge $MAX_CORE ]; then
        CORENUM=0
    fi
done

wait

echo "All processing jobs have completed."

exit 0
