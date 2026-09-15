#!/bin/bash
# Verifies PAM auth for a user using the password stored in the root-only
# password file, without ever printing the password itself.
set -euo pipefail
user="$1"
pw="$(grep "^${user}:" /root/tljh_initial_passwords.txt | cut -d: -f2-)"
python3 -c "
import pamela, sys
user = sys.argv[1]
pw = sys.argv[2]
try:
    pamela.authenticate(user, pw)
    print('PAM auth OK for', user)
except Exception as e:
    print('PAM auth FAILED for', user, '->', e)
" "$user" "$pw"
