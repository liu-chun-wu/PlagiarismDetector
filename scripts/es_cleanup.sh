#!/bin/bash
echo "== 開始清理 $(date) =="

# 列出所有索引並刪除（謹慎操作）
indices=$(curl -s http://elasticsearch:9200/_cat/indices?h=index)
for index in $indices; do
  echo "刪除索引：$index"
  curl -X DELETE "http://elasticsearch:9200/$index"
done

echo "== 清理完成 $(date) =="
