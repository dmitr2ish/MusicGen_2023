#!/bin/bash

while true; do
    read -p "Continue? yes/no: " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please, answer yes or no.";;
    esac
done

# Source the environment setup script
source ./setenv.sh

# Run the main Python script
streamlit run src/music_gen_app.py

# Deactivation is not strictly necessary for a script, as the environment changes
# do not persist when the script finishes
echo "Script execution finished."
