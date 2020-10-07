#!/usr/bin/env bash

cd "$(dirname "${0}")"

curl -L https://www.dropbox.com/s/efmencd3ihtsnr5/artifact.tgz?dl=0 | tar xz
curl -L https://www.dropbox.com/s/jewld1j81ovh4wg/bhive.tgz?dl=0 | tar xz
