# MELD Automated Builds

We have an automated build system in place for MELD.
It automatically builds and creates conda packages
for MELD. We are using travis-ci as our build
platform. All build results are available at
<https://travis-ci.org/maccallumlab>.

There are two sets of builds:
- release builds
- `test` builds

## Conda Packages

We use the [anaconda](http://anaconda.org) package
manager. The packages end up in two places, depending
on if they are release or test builds.
- release builds are found at <http://anaconda.org/omnia>
- test builds are found at <http://anaconda.org/maccallum_lab>

To use these packages, you must first add a channel and then
load the appropriate package. For example:
```
conda config --add channels omnia
conda install meld
```

**WARNING**: anaconda currently has no way to specify that
packages are mutually exclusive. So, it's possible to 
install both release and test versions, which will
cause problems.

## Release Builds

The release builds are intended for wide distribution
and use.

The release builds are triggered when the
[omnia-md/conda-recipes](http://github.com/omnia-md/conda-recipes)
repository is modified. This repository specifies a
specific git revision that is uploaded to
the [omnia channel](https://anaconda.org/omnia).

## Test Builds

The test builds are intended for testing of new features.
They are bleeding-edge builds with all of the latest
code, but may contain bugs.

The test builds are triggered every time there is a
commit or pull request to `maccallumlab/meld`.

For the `master` branch Commits are built and uploaded as `meld-test` to the 
[maccallum_lab](https://anaconda.org/maccallum_lab)
anaconda channel.

For pull requests and other branches, the build is performed,
but no packages are uploaded. Pull requests will not be merged
until the build passes.
