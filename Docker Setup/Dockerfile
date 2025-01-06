# Use the official Amazon Linux 2 image
FROM amazonlinux:2

# Install required dependencies including Perl
RUN yum install -y \
    gcc \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    zlib-devel \
    wget \
    make \
    tar \
    gzip \
    xz \
    libffi-devel \
    perl

# Verify Perl installation
RUN perl -v

# Download and build OpenSSL
RUN cd /usr/src && \
    wget https://www.openssl.org/source/openssl-1.1.1l.tar.gz && \
    tar xzf openssl-1.1.1l.tar.gz && \
    cd openssl-1.1.1l && \
    ./config --prefix=/usr/local/ssl --openssldir=/usr/local/ssl shared zlib && \
    make && \
    make install

# Update the system library path to include the newly installed OpenSSL libraries
ENV LD_LIBRARY_PATH /usr/local/ssl/lib:$LD_LIBRARY_PATH
ENV CPATH /usr/local/ssl/include:$CPATH
ENV LIBRARY_PATH /usr/local/ssl/lib:$LIBRARY_PATH

# Download and install Python 3.10
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar xzf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations --with-openssl=/usr/local/ssl && \
    make altinstall

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py
# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Command to run your application
CMD ["python3.10", "app.py"]