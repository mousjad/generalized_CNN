#!/bin/sh
ID_RSA=C:\Users\mojad.MECA\.ssh\id_ed25519
exec ssh -o StrictHostKeyChecking=no -i $ID_RSA "$@"