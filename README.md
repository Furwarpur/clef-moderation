# clef2

Clef-Flash 9Bでライブコメントの掲載可否を判定するWebSocket workerです。静的ルールとモデル推論を組み合わせ、`allow`・`block`・`review` のいずれかを返します。理由分類・該当箇所抽出の追加推論やHTTP推論APIはありません。

## 判定の流れ

1. コメントID・チャンネル・本文を検証します。本文は1〜20文字で、空白だけの文字列、制御文字、サロゲート文字は受け付けません。
2. 静的ルールに一致したコメントは、モデル推論を行わず `block` を返します。
3. それ以外は、コメント原文と掲載基準を公式エンコーダーで入力に変換し、公式の判定headを使った1回の推論で掲載可否を決めます。

| 判定 | 意味 |
| --- | --- |
| `allow` | 掲載基準で許可されるコメント |
| `block` | 掲載基準の禁止事項に該当するコメント |
| `review` | 情報・文脈が不足して判断できないコメント、または正常に判定できなかったコメント |

掲載基準は [policy-live-v1.md](policy-live-v1.md)、静的ルールの語句は [live-policy-v1.json](live-policy-v1.json) に固定しています。コメント内の命令・権限主張・ポリシー変更には従わないよう、判定質問に明記しています。

## 静的判定

語句照合ではNFKC正規化・大文字小文字の統一・カタカナのひらがな化を行い、不可視の書式文字・空白・句読点を除去します。URL判定では別途URLパターンを照合します。この正規化は静的判定用で、モデルにはコメント原文を渡します。

一致したカテゴリを、次の順序ですべて返します。

| カテゴリ | 対象 |
| --- | --- |
| `tsukuba_person` | 指定された筑波大学関係者の名前・表記 |
| `imperial_topic` | 指定された皇室関連の語句 |
| `nuclear_topic` | 指定された核実験関連の語句 |
| `url` | URLやドメインのパターン |

## 入出力

`moderation.task` をWebSocketで受信します。`comment_id` は空でないUTF-8文字列、`channel_id` は `uni`・`1A`・`kaikan` のいずれかです。

```json
{"type":"moderation.task","comment_id":"example-1","channel_id":"uni","text":"がんばれ"}
```

結果は次の形式です。モデルによる判定では `categories` は常に空配列で、静的判定による `block` の場合だけ一致したカテゴリを入れます。理由・該当箇所・確信度は返しません。

```json
{"type":"moderation.result","comment_id":"example-1","worker_id":"clef","decision":"allow","categories":[]}
```

接続認証には、WebSocketハンドシェイクの `Sec-WebSocket-Protocol` に `bearer.<token>` を指定します。トークンをURLには含めません。

## 推論構成と制限

Python 3.12、ROCm対応のPyTorchとvLLMを使い、GPUは `gfx1151` を対象とします。モデル・プラグイン・主要パッケージの版とファイルのハッシュは [manifest.json](manifest.json) に固定しています。

vLLMエンジンを常駐させ、FP16、テンソル並列1、データ並列1で処理します。workerはFIFOキューから1件ずつ判定し、prefix cacheは使用しません。入力は掲載基準を含めて最大2048トークンで、超過時は切り詰めずにエラーとして扱います。

## 設定

| 環境変数 | 未指定時 | 用途 |
| --- | --- | --- |
| `MODERATION_URL` | 必須 | バックエンドのWebSocket接続先。パスは `/ws/moderation`。ローカル接続以外は `wss` 必須 |
| `MODERATION_TOKEN` | 必須 | バックエンドと共通の認証トークン |
| `CLEF_INFERENCE_TIMEOUT_SECONDS` | `30` | 1件の推論時間上限（秒）。モデル読み込み・キュー待ちは除く |
| `CLEF_QUEUE_CAPACITY` | `16` | 実行中の1件を除く待機件数 |
| `PYTORCH_TUNABLEOP_ENABLED` | `1` | TunableOpと保存済み調整結果の使用。`0` または `1` |
| `PYTORCH_TUNABLEOP_TUNING` | `0` | 追加のオンライン調整。`0` または `1` |
| `PYTORCH_TUNABLEOP_FILENAME` | `runtime/tunableop.csv` | 調整結果のCSV。相対パスはclef2ディレクトリ基準。絶対パスも指定でき、保存先はclef2/runtime配下に限定 |

不正な設定値、固定ファイルのハッシュ不一致、調整CSVの形式・実行環境の不一致では起動しません。

## TunableOpによる最適化

オンライン調整はモデルの重みを学習する処理ではなく、実際に使う行列演算についてGPUカーネルの実行時間を比較し、速い実装を選ぶ処理です。初めて調整する条件では判定やモデル読み込みに時間がかかる場合があります。

TunableOpと追加調整が両方有効な場合、GPU側の推論完了後に調整結果が増えていればCSVを一時ファイルから置換して保存します。保存に失敗した場合は既存CSVを保持し、次回に再試行します。原本 `runtime/tunableop-original.csv` は更新しません。中断時の未保存分は引き継げません。

`PYTORCH_TUNABLEOP_ENABLED=1` のまま `PYTORCH_TUNABLEOP_TUNING=0` にすると、新たな調整を行わず、同じCSVに保存された結果を再利用できます。`PYTORCH_TUNABLEOP_ENABLED=0` では保存済み結果も使用しません。オフライン調整用の未調整値記録は無効に固定しています。

## 判定できない場合の動作

正常なコメントIDを取得できたタスクでは、不正入力・回復可能な推論失敗・不正出力・キュー満杯時に `review`、`categories: []` を返します。タイムアウトやGPU異常では、処理中・待機中のタスクへの `review` 送信を全体で最大5秒試み、終了します。

通信断・不正JSON・ID不明・送信失敗では返信できず、バックエンドにpendingが残る場合があります。結果の独自再送やキューの永続化は行いません。

通常ログはJSONで、接続状態・ID・処理時間・結果・エラー種別を記録します。トークン、コメント原文、例外本文は記録しません。

## 実装の構成

`worker.py` が接続・キュー・返信、`model.py` が公式エンコーダー・公式判定head・vLLM、`policy.py` が掲載基準と静的判定を担当します。`clef_kernel_hooks.py` は固定プラグインがGPUプロセスから調整処理を読み込むための入口です。
