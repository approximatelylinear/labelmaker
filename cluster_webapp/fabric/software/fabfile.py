
import os
from fabric.api import env, local, settings, abort, run, cd, sudo, put
from fabric.contrib.console import confirm
from fabric.context_managers import prefix

env.user = 'mberends'
# env.sudo_user = 'mattb'
env.hosts = [
    # '167.246.44.210', # na1plbapl1412v.naext.pubgroupeext.net
    '10.3.192.102',
]
env.password = 'iLwb!T5*' # mattb2014
env.no_keys = True


PATH_HOME = '/home/mberends'
PATH_APP = '/usr/local/app'
PATH_APP_SRC = os.path.join(PATH_APP, 'src')
PATH_APP_GIT = os.path.join(PATH_APP, 'git')

#   Utilities
def ungzip(path, fname):
    with cd(path):
        cmd = 'tar xzvf {}'.format(fname)
        run(cmd)

def unbzip(path, fname):
    with cd(path):
        cmd = 'tar xjvf {}'.format(fname)
        run(cmd)

def unzip(path, fname):
    with cd(path):
        cmd = 'unzip {}'.format(fname)
        run(cmd)

def unxzip(path, fname):
    #   Assume fname in form: ___.tar.xz
    with cd(path):
        cmd = 'xz -d {}'.format(fname)
        run(cmd)
        cmd = 'tar -xvf {}.tar'.format(fname[:-3])
        run(cmd)

def checkout_bare_repo(path_work, path_repo):
    cmd = "git --work-tree={w} --git-dir={r} checkout -f".format(
        w=path_work, r=path_repo
    )
    run(cmd)

def add_envvar(name, value):
    cmds = [
        'export export {n}={v}'.format(n=name, v=value),
        'echo "export {n}={v}" >> ${{HOME}}/.bash_profile'.format(
            n=name, v=value
        ),
    ]
    for cmd in cmds:
        run(cmd)

def basic_install(version, flags, fname=None, url=None, **kwargs):
    cmd = 'mkdir -p {}'.format(PATH_APP_SRC)
    run(cmd)
    if url:
        fname = url.rsplit('/', 1)[-1]
        cmd = "curl {} -o {}".format(url, os.path.join(PATH_APP_SRC, fname))
        run(cmd)
    else:
        if fname is None:
            fname = '{}.tar.gz'.format(version)
        # put(
        #     local_path=os.path.join(THIS_DIR, fname),
        #     remote_path=PATH_APP_SRC
        # )
    ungzip(PATH_APP_SRC, fname)
    with cd(os.path.join(PATH_APP_SRC, version)):
        cmds = [
            "make clean",
            './configure {}'.format(' '.join(flags)),
            "make",
            "make check",
            "make install",
            "make check-install",
        ]
        for cmd in cmds:
            run(cmd)

#   --------------------------------------

def main():
    add_linalg()
    add_deps_pytables()
    add_python()
    add_clustering_app()


def add_clustering_app():
    add_app('cluster_webapp', 'dev')


def checkout_cluster_app():
    #   Check out main and nested repos.
    bp_work = '/usr/local/app/cluster_webapp'
    bp_repo = '/usr/local/app/git'
    checkout_bare_repo(
        os.path.join(bp_work, 'cluster_webapp'),
        os.path.join(bp_repo, 'cluster_webapp')
    )
    bp_work = os.path.join(bp_work, 'lib')
    checkout_bare_repo(
        os.path.join(bp_work, 'clustering'),
        os.path.join(bp_repo, 'clustering')
    )
    checkout_bare_repo(
        os.path.join(bp_work, 'text_processing'),
        os.path.join(bp_repo, 'text_processing')
    )


def add_devtools():
    # #   Update system
    cmd = 'yum -y update'
    sudo(cmd)
    # #   Install devtools
    cmd = 'yum groupinstall -y "Development tools"'
    sudo(cmd)
    #   Install other necesary pkgs.
    pkgs = [
        "yum-utils",
        "zlib-devel",
        "openssl-devel",
        "sqlite-devel",
        "bzip2-devel",
        "xz-libs",
        "curl-devel"
        "httpd-devel",
        "apr-devel",
        "apr-util-devel",
        "readline-devel",
        "libffi-devel",
        "links", # apachectl dep
        # "openmpi",
        # "openmpi-devel",
        "lzo",
        "lzo-devel",
        "libpng",
        "libpng-devel",
        #   Linear algebra libraries for scipy etc.
        #       Are these the right pkg names?
        "gfortran", # gcc-gfortran.x86_64, libgfortran.x86_64 ??
        "libopenblas-devel",
        "liblapack-devel", # lapack.x86_64
    ]
    cmd = 'yum install -y {}'.format(' '.join(pkgs))
    sudo(cmd)


def add_python():
    add_python_27()
    add_setuptools()
    add_pip()
    add_virtualenv()


def clone_app(name, name_remote='origin'):
    cmd = "git clone --bare {name} {name}.git".format(name=name)
    local(cmd)
    cmd = "mkdir -p {}".format(PATH_APP_GIT)
    put(
        "{}.git".format(name),
        PATH_APP_GIT
    )
    with cd(os.path.join(PATH_APP_GIT, "{}.git".format(name))):
        cmd = "git init --bare --shared"
        run(cmd)
    cmd = "git remote add {r} {u}@{h}:{p}.git".format(
        r=name_remote, u=env.user,
        h=env.hosts[0], p=os.path.join(PATH_APP_GIT, name),
    )
    local(cmd)
    cmd = "git push {r} master".format(name_remote)
    local(cmd)


def add_app(name, name_remote='origin'):
    path = os.path.join(PATH_APP, name)
    cmd = 'mkdir -p {}'.format(path)
    run(cmd)
    clone_app(name, name_remote)
    #   Create a virtualenv inside the app directory
    #   (Assumes that the pip requirements exist in that dir.)
    create_virtualenv(path, name)


def add_python_27():
    #   source: https://www.python.org/ftp/python/2.7.8/Python-2.7.8.tar.xz
    version_python = 'Python-2.7.8'
    fname_python = '{}.tar.xz'.format(version_python)
    #   Get deps
    cmd = "yum-builddep python-matplotlib"
    sudo(cmd)
    put(
        local_path=os.path.join(THIS_DIR, fname_python),
        remote_path="${HOME}/src"
    )
    unxzip('${HOME}/src', fname_python)
    with cd('${{HOME}}/src/{}'.format(version_python)):
        #   Configure
        cmd = './configure --prefix=/usr/local'
        run(cmd)
        cmd = 'make'
        run(cmd)
        cmd = 'make altinstall'
        sudo(cmd)


def add_pip():
    version_python = '2.7'
    version_pip = 'pip-1.5.6'
    fname_pip = '{}.tar.gz'.format(version_pip)
    put(
        local_path=os.path.join(THIS_DIR, fname_pip),
        remote_path='${HOME}/src'
    )
    ungzip('${HOME}/src', fname_pip)
    with cd('${HOME}/src/pip-*/'):
        cmd = '/usr/local/bin/python{} setup.py install'.format(version_python)
        sudo(cmd)

#   casperjs
"""
git clone git://github.com/n1k0/casperjs.git
cd casperjs
ln -sf `pwd`/bin/casperjs /usr/local/bin/casperjs
"""

def add_setuptools():
    #   https://pypi.python.org/packages/source/s/setuptools/setuptools-6.0.2.zip
    version_python = '2.7'
    version_setuptools = 'setuptools-6.0.2'
    fname_setuptools = '{}.zip'.format(version_setuptools)
    put(
        local_path=os.path.join(THIS_DIR, fname_setuptools),
        remote_path='${HOME}/src'
    )
    unzip('${HOME}/src', fname_setuptools)
    with cd('${{HOME}}/src/{}'.format(fname_setuptools)):
        cmd = '/usr/local/bin/python{} setup.py install'.format(version_python)
        sudo(cmd)


def add_virtualenv():
    cmd = 'pip2.7 install virtualenv'
    sudo(cmd)


def create_virtualenv(path, name):
    fname_req = 'pipreq.txt'
    path_src = 'pysrc'
    with cd(path):
        cmds = [
            'virtualenv {}'.format(name),
            "mkdir -p {}".format(path_src),
            "pip install --requirement={} --download={}".format(
                fname_req, path_src
            )
        for cmd in cmds:
            run(cmd)
        with prefix('source {}/bin/activate'.format(name)):
            cmd = "pip2.7 install --no-index --find-links={} -r {}".format(
                path_src, fname_req
            )
            run(cmd)


def download_pkgs(pkgs):
    fname_req = 'pipreq.txt'
    path_src = 'pysrc'
    with cd(path):
        requirements = "\n".join(pkgs)
        cmds = [
            "mkdir -p {}".format(path),
            'echo "{}" > {}'.format(requirements, path_req),
            cmd = "pip install --requirement={} --download={}".format(
                fname_req, path_src
            )
        ]
        for cmd in cmds:
            run(cmd)


def install_pkgs(path):
    fname_req = 'pipreq.txt'
    path_src = 'pysrc'
    with cd(path):
        cmd = "pip2.7 install --no-index --find-links={} -r {}".format(
            path_src, fname_req
        )
        run(cmd)


#   ---------------------------------------------------

"""
If you ever happen to want to link against installed libraries
in a given directory, LIBDIR, you must either use libtool, and
specify the full pathname of the library, or use the `-LLIBDIR'
flag during linking and do at least one of the following:
   - add LIBDIR to the `LD_LIBRARY_PATH' environment variable
     during execution
   - add LIBDIR to the `LD_RUN_PATH' environment variable
     during linking
   - use the `-Wl,--rpath -Wl,LIBDIR' linker flag
   - have your system administrator add LIBDIR to `/etc/ld.so.conf'
"""

def add_deps_pytables():
    add_szip()
    add_lzo()
    add_hdf5()


def add_szip():
    url = 'http://www.hdfgroup.org/ftp/lib-external/szip/2.1/src/szip-2.1.tar.gz'
    version = url.rsplit('/', 1)[-1][:-len('.tar.gz')]
    path_szip = '{}/szip'.format(PATH_APP)
    basic_install(
        version,
        flags=["--prefix='{}'".format(path_szip)],
        url=url,
    )
    add_envvar('SZIP_DIR', path_szip)
    add_envvar('SZIP_LIB', os.path.join(path_szip, 'lib'))
    add_envvar('SZIP_INCLUDE', os.path.join(path_szip, 'include'))
    #   Add paths to system path
    paths = ':'.join(['${SZIP_DIR}', '${SZIP_LIB}', '${SZIP_INCLUDE}'])
    cmd = 'echo "export PATH=${{PATH}}:{}" >> ${{HOME}}/.bash_profile'.format(paths)
    run(cmd)


def add_hdf5():
    """
    tar zxf
    cd hdf5-X.Y.Z
    ./configure --prefix=/usr/local/hdf5 <more configure_flags>
    make clean
    make
    make check                # run test suite.
    make install
    make check-install        # verify installation.
    """
    url = 'http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.13.tar.gz'
    version = url.rsplit('/', 1)[-1][:-len('.tar.gz')]
    path_szip = os.path.join(PATH_APP, 'szip')
    path_hdf5 = os.path.join(PATH_APP, 'hdf5')
    basic_install(
        version,
        flags=[
            "--prefix='{}'".format(path_hdf5),
            "--with-szlib={}".format(path_szip),
            "--enable-cxx", # C++ interface
        ],
        # url=url,
    )
    # add_envvar('HDF5_DIR', path_hdf5)
    # add_envvar('HDF5_LIB', os.path.join(path_hdf5, 'lib'))
    # add_envvar('HDF5_INCLUDE', os.path.join(path_hdf5, 'include'))
    # #   Add paths to system path
    # paths = ':'.join(['${HDF5_DIR}', '${HDF5_LIB}', '${HDF5_INCLUDE}'])
    # cmd = 'echo "export PATH=${{PATH}}:{}" >> ${{HOME}}/.bash_profile'.format(paths)
    # run(cmd)
    # cmd = (
    #     'echo "export LD_LIBRARY_PATH=${{LD_LIBRARY_PATH}}:{}\n
    #     'export LD_RUN_PATH=${{LD_LIBRARY_PATH}}:{}" '
    #     '>> ${{HOME}}/.bash_profile'
    # ).format(os.path.join(path_hdf5, 'lib'))


def add_lzo():
    url = 'http://www.oberhumer.com/opensource/lzo/download/lzo-2.08.tar.gz'
    version = url.rsplit('/', 1)[-1][:-len('.tar.gz')]
    path_lzo = os.path.join(PATH_APP, 'lzo')
    basic_install(
        version,
        flags=["--prefix='{}'".format(path_lzo)],
        url=url,
    )
    add_envvar('LZO_DIR', path_lzo)
    add_envvar('LZO_LIB', os.path.join(path_lzo, 'lib'))
    add_envvar('LZO_INCLUDE', os.path.join(path_lzo, 'include'))
    #   Add paths to system path
    paths = ':'.join(['${LZO_DIR}', '${LZO_LIB}', '${LZO_INCLUDE}'])
    cmd = 'echo "export PATH=${{PATH}}:{}" >> ${{HOME}}/.bash_profile'.format(paths)
    run(cmd)


def add_linalg():
    add_blas()
    add_lapack()
    add_atlas()


def add_lapack():
    put_lapack()
    compile_lapack()


def add_blas():
    fname_blas = 'blas.tgz'
    cmd = 'mkdir -p ${HOME}/src'
    run(cmd)
    put(
        local_path=os.path.join(THIS_DIR, fname_blas),
        remote_path='${HOME}/src'
    )
    ungzip('${HOME}/src', fname_blas)
    with cd('${HOME}/src/BLAS'):
        cmds = [
            #   Compile BLAS with gfortran
            'gfortran -O3 -std=legacy -m64 -fno-second-underscore -fPIC -c *.f',
            #   Combine the output files into an archive
            'ar r libfblas.a *.o',
            #   Clean up the original output files
            'rm -rf *.o',
            #   Index the archive
            'ranlib libfblas.a',
            #   Create BLAS env variable and add to profile.
            'export BLAS=${HOME}/src/BLAS/libfblas.a',
            'echo "export BLAS=${HOME}/src/BLAS/libfblas.a" >> ${HOME}/.bash_profile',
        ]
        for cmd in cmds:
            run(cmd)
    #   Note: The archive is the right BLAS path
    #       export BLAS=${HOME}/src/BLAS/libfblas.a


def put_lapack():
    fname_lapack = 'lapack.tgz'
    #   Source: http://www.netlib.org/lapack/lapack.tgz
    cmd = 'mkdir -p ${HOME}/src'
    run(cmd)
    put(
        local_path=os.path.join(THIS_DIR, fname_lapack),
        remote_path='${HOME}/src'
    )
    ungzip('${HOME}/src', fname_lapack)
    with cd('${HOME}/src/lapack-*'):
        cmd = 'cp INSTALL/make.inc.gfortran make.inc'
        run(cmd)
        #   Change the make.inc file:
        #   vim make.inc
        #       OPTS     = -O2 -fPIC -m64
        #       NOOPT = -O0 -fPIC -m64
        msg = """
Change "make.inc":
    vim make.inc

        OPTS     = -O2 -fPIC -m64
        NOOPT = -O0 -fPIC -m64

        ---
        Done?
        """
        print msg
        done = raw_input()


def compile_lapack():
    fname_lapack = 'lapack-3.5.0'
    with cd('${HOME}/src/lapack-*'):
        cmd = 'make lapacklib && make clean'
        run(cmd)
        cmd = 'export LAPACK=${{HOME}}/src/{}'.format(fname_lapack)
        run(cmd)
        cmd = 'echo "export LAPACK=${{HOME}}/src/{}" >> ${{HOME}}/.bash_profile'.format(
            fname_lapack
        )
        run(cmd)
        #   Note: The lapack root directory is the right LAPACK path
        #       export LAPACK=${HOME}/src/lapack-3.5.0

def add_atlas():
    fname_atlas = 'atlas3.10.2.tar.bz2'
    fname_lapack = 'lapack-3.5.0'
    path_lapack = '${{HOME}}/src/{}'.format(fname_lapack)
    threaded = False
    #   Source: http://www.netlib.org/lapack/lapack.tgz
    cmd = 'mkdir -p ${HOME}/src'
    run(cmd)
    put(
        local_path=os.path.join(THIS_DIR, fname_atlas),
        remote_path='${HOME}/src'
    )
    unbzip('${HOME}/src', fname_atlas)
    with cd('${HOME}/src/ATLAS'):
        cmd = 'mkdir ATLAS_x86_64_Linux'
    with cd('${HOME}/src/ATLAS/ATLAS_x86_64_Linux'):
        cmd = '../configure -Fa alg -fPIC --with-netlib-lapack={}/liblapack.a'.format(
            path_lapack
        )
        run(cmd)
        cmd = 'make'
        run(cmd)
    with cd('${HOME}/src/ATLAS/ATLAS_x86_64_Linux/lib'):
        if threaded:
            cmd = 'make ptshared' # for threaded libraries
        else:
            cmd = 'make shared' # for sequential libraries
        run(cmd)



def add_mallet():
    uri_mallet = 'https://github.com/mimno/Mallet.git'
    fname_ant = 'apache-ant-1.9.4'
    with cd(PATH_APP):
        cmd = 'git clone {}'.format(uri_mallet)
        sudo(cmd)
    path_mallet = os.path.join(PATH_APP, fname_ant)
    path_ant = os.path.join(PATH_APP, fname_ant)
    with cd(path_mallet):
        #   Compile the Mallet code with Ant
        cmd = os.path.join(path_ant, 'bin', 'ant')
        sudo(cmd)
        #   Do I need these?
        cmd = 'javac'
        sudo(cmd)
        cmd = '{} jar'.format(os.path.join(path_ant, 'bin', 'ant'))
        sudo(cmd)
        #   Add "MALLET_HOME" to the current user's profile.
        cmd = 'echo "MALLET_HOME={}/Mallet" >> ${{HOME}}/.bash_profile'.format(
            PATH_APP
        )
        run(cmd)
