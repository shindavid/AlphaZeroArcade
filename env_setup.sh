# Get the directory where the env_setup.sh script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

ENV_VARS_SCRIPT="$SCRIPT_DIR/.env.sh"
SETUP_WIZARD_SCRIPT="$SCRIPT_DIR/setup_wizard.py"

if [ -e "$ENV_VARS_SCRIPT" ]; then
    source "$ENV_VARS_SCRIPT"
else
    python $SETUP_WIZARD_SCRIPT

    if [ -e "$ENV_VARS_SCRIPT" ]; then
        source "$ENV_VARS_SCRIPT"
    fi
fi

# Validate that md5sum of environment.yml file matches the one in .environment.yml.md5:
MD5_MATCH=0
if [ -e "$SCRIPT_DIR/.environment.yml.md5" ]; then
    MD5_SUM=$(md5sum "$SCRIPT_DIR/environment.yml")
    MD5_SUM=$(echo $MD5_SUM | cut -d ' ' -f 1)
    EXPECTED_MD5_SUM=$(cat "$SCRIPT_DIR/.environment.yml.md5")
    if [ "$MD5_SUM" == "$EXPECTED_MD5_SUM" ]; then
        MD5_MATCH=1
    fi
fi

if [ $MD5_MATCH -eq 0 ]; then
    echo "Your conda environment may be out of date. Consider running py/update_conda_env.py"
fi