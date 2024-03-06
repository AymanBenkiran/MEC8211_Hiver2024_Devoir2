#!/bin/bash
#set -ex
# But: ce script efface les lignes superflus dans les fichiers solutions
# exportees de COMSOL.

# Execution call: ./cleanup_solution_files.sh

#---------------------------------------------------------------------------------
# Messages
#---------------------------------------------------------------------------------
  USAGE="Usage: ./cleanup_solution_files.sh"

#---------------------------------------------------------------------------------
# Script:
#---------------------------------------------------------------------------------
# Folder with solutions
FOLDER="./comsol_solutions/"
# Get a list of the csv files
FILES_LIST=$(ls $FOLDER)
# Define the lines to delete
LINES_TO_DELETE=$(echo {1..7})

# Create a temporary file
TEMP_FILE=$(mktemp)

# Loop over files
# shellcheck disable=SC2068
for FILE in ${FILES_LIST[@]}
do
  FILE=$FOLDER$FILE
  if grep -q "%" "$FILE"; then
    # Copy contents of original file to temporary file, omitting the lines to be erased
    awk -v LINES="${LINES_TO_DELETE[*]}" '
      BEGIN {
          split(LINES, REMOVE_ARR)
          for (i in REMOVE_ARR) {
              REMOVE_LINES[REMOVE_ARR[i]]
          }
      }
      ! (NR in REMOVE_LINES)
      ' "$FILE" > "$TEMP_FILE"

    # Remove "% " at the beginning of the header line
    sed -i '' "${1}s/% //g" "$TEMP_FILE"

    # Replace temporary file
    mv "$TEMP_FILE" "$FILE"
  fi
done