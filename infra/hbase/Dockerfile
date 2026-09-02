FROM eclipse-temurin:11-jre-jammy

ARG HBASE_VERSION=2.5.15
ARG HBASE_SHA512=60a07d97ec68203c3494637f4abcb6fdc2fcc75c4c1b88b80d5d8616640e2a57a5193067849f677d441cae558ede69ee4e0ce489816cbcb9117db268fc316ae5

RUN apt-get update && \
    apt-get install --yes --no-install-recommends curl procps && \
    rm -rf /var/lib/apt/lists/* && \
    curl --fail --location --silent --show-error \
        "https://downloads.apache.org/hbase/${HBASE_VERSION}/hbase-${HBASE_VERSION}-bin.tar.gz" \
        --output /tmp/hbase.tar.gz && \
    echo "${HBASE_SHA512}  /tmp/hbase.tar.gz" | sha512sum --check - && \
    mkdir -p /opt/hbase && \
    tar --extract --gzip --file /tmp/hbase.tar.gz --strip-components=1 --directory /opt/hbase && \
    rm /tmp/hbase.tar.gz 

ENV HBASE_HOME=/opt/hbase \
    PATH=/opt/hbase/bin:$PATH

COPY hbase-site.xml /opt/hbase/conf/hbase-site.xml
COPY entrypoint.sh /usr/local/bin/hbase-entrypoint

RUN chmod +x /usr/local/bin/hbase-entrypoint && mkdir -p /data/hbase /data/zookeeper

VOLUME ["/data"]
EXPOSE 9090 9095 16010

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=12 \
    CMD curl --fail --silent http://127.0.0.1:16010/master-status >/dev/null || exit 1

ENTRYPOINT [ "/usr/local/bin/hbase-entrypoint" ]