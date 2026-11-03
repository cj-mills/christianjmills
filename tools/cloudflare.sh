#!/bin/sh
# Runs a command with Cloudflare credentials injected by 1Password, so the token never sits on
# disk or in the shell's history:
#
#   sh tools/cloudflare.sh npx wrangler whoami
#
# The env file holds 1Password references (op://…), not secrets. It names a private vault and
# item, so it lives outside this public repo, by default at
# ~/.config/christianjmills/cloudflare.env (override with CJM_CLOUDFLARE_ENV):
#
#   CLOUDFLARE_API_TOKEN=op://<vault>/<item>/<field>
#   CLOUDFLARE_ACCOUNT_ID=op://<vault>/<item>/<field>
set -eu
env_file="${CJM_CLOUDFLARE_ENV:-$HOME/.config/christianjmills/cloudflare.env}"
if [ ! -f "$env_file" ]; then
  echo "cloudflare.sh: no env file at $env_file (see the comment at the top of this script)" >&2
  exit 2
fi
export WRANGLER_SEND_METRICS=false
exec op run --env-file="$env_file" -- "$@"
