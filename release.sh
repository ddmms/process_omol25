#!/usr/bin/env bash

version=$1

sed -i.bak "s;^version =.*;version = \"$version\";g" pyproject.toml && rm pyproject.toml.bak
sed -i.bak "s;^version =.*;version = \"$version\";g" docs/conf.py && rm docs/conf.py.bak
sed -i.bak "s;^release =.*;release = \"$version\";g" docs/conf.py && rm docs/conf.py.bak

git add pyproject.toml
git add docs/conf.py
git commit -m "bump version for release $version"
git tag -f -a v$version -m "release $version"
git push --force-with-lease
git push --tags
