FROM fluent/fluentd:v1.16-1

USER root

RUN gem install fluent-plugin-elasticsearch

COPY fluent.conf /fluentd/etc/
