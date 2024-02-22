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
