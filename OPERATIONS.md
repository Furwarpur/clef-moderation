# clef2 運用手順・検証記録

モデルの仕様と判定動作は [README.md](README.md) を参照してください。以下の稼働状況・検証結果は、各項目に記載した日時の記録です。

## 環境と起動

Python 3.12の検証済み環境・モデル・プラグインは `runtime/` に配置済みです。`manifest.json` に版とハッシュを記録しています。移設後に仮想環境の起動スクリプトもこのディレクトリへ向け直してあり、旧ディレクトリは実行時に参照しません。`runtime/` と `.env` はGit管理対象外です。Gitだけで別の場所へ展開する場合は、別途これらの成果物が必要です。

```sh
runtime/venv/bin/python -B worker.py --check
runtime/venv/bin/python -B worker.py --check --verify-model
runtime/venv/bin/python -B -m unittest discover -s tests -v
```

最初のコマンドは固定ファイル・主要パッケージ・CSV形式を確認します。`--verify-model` は大容量の重みもハッシュ照合します。いずれもGPUエンジンやバックエンド接続を起動しません。GPU環境とのCSVの互換性はエンジン起動時にも検証します。

`.env` は既存workerの最新接続情報を引き継ぎ、権限0600で配置済みです。トークンは仕様書やこの文書には記載しません。手動実行時は以下のように読み込みます。

```sh
set -a
. ./.env
set +a
runtime/venv/bin/python -B worker.py
```

既存GPU APIが11634番で動いている場合は起動を拒否します。実GPU検証・常駐切り替えは、停止範囲と時間を確認してから行います。

## 環境変数の変更

各変数の意味と未指定時の値は [README.md](README.md)、設定例は [.env.example](.env.example) を参照してください。

環境変数の変更を反映するには再起動が必要です。`/home/fwpw/projects/live/clef2/.env` を編集後、clef2を動かしている計算機上で、ユーザー `fwpw` として次のコマンドを実行してください。実行ディレクトリはどこでも構いません。

```sh
systemctl --user restart clef2-worker.service
```

再起動時にsystemdが `.env` を読み直すため、`.env` の変更だけなら `source` や `daemon-reload` は不要です。不正値では起動しません。オフライン調整用の未調整値記録は無効に固定しています。

## オンライン調整と本番当日

本番前は `PYTORCH_TUNABLEOP_ENABLED=1`、`PYTORCH_TUNABLEOP_TUNING=1` とします。未調整の条件を測定し、GPU側の推論完了後に結果が増えていればCSVを一時ファイルから置換して保存します。保存失敗時は既存CSVを保持してエラーを記録し、次回に再試行します。検証済み原本 `runtime/tunableop-original.csv` は更新しません。

初めて調整する条件では遅延が伸びます。今回、測定上限を1ms・1回に制限しても初回判定に約95秒かかったため、調整時は `CLEF_INFERENCE_TIMEOUT_SECONDS` を600秒等へ延長してください。サンプルでは600秒にしています。通常の候補測定上限での所要時間は未測定で、600秒以内の完了を保証しません。中断した調整の未保存分は引き継げません。

当日はサービス停止後に `PYTORCH_TUNABLEOP_TUNING=0`、`CLEF_INFERENCE_TIMEOUT_SECONDS=30` へ変更し、同じCSVで再起動します。`PYTORCH_TUNABLEOP_ENABLED` は1のままにします。停止した状態で本番用ハッシュを保存すると、追加調整無効の起動時に照合します。

```sh
runtime/venv/bin/python -B - <<'PY'
from pathlib import Path
from model import digest
path = Path('runtime/tunableop.csv')
path.with_suffix('.csv.sha256').write_text(digest(path) + '\n')
PY
```

再調整後は停止した状態でハッシュを記録し直します。CSVの形式・実行環境の不一致では起動しません。

## 実GPU比較と常駐切り替え

停止の確認を取ってから、旧worker・旧Web画面・旧GPU APIをすべて停止します。比較元は検証済みの旧起動コードを使い、HTTPサーバー起動だけを置き換えて判定処理を直接呼び出します。比較元とclef2を直列に起動し、HTTP APIを公開せずに測定します。両方でオンライン調整を無効にし、同一CSVを使用します。

```sh
runtime/venv/bin/python -B tests/compare_gpu.py baseline
runtime/venv/bin/python -B tests/compare_gpu.py compare
```

結果は `runtime/comparison-baseline.json` と `runtime/comparison.json` に保存します。非静的24件をウォームアップ後に各3回処理し、判定の全件一致と掲載判定中央値110%以内を確認します。最大値と95パーセンタイル、各エンジン起動後の最初のリクエストも記録します。生のコメント本文は比較結果に保存しません。clef2ではローカルWebSocketを通した49件の判定と、不正入力のreview返信も確認します。

オンライン調整の保存と再起動後の再利用は、専用CSVを使って次の順で検証します。検証時間を制限するため、この試験だけ各候補の測定上限を5ms・4回にしています。運用の作業CSVと原本は変更しません。

```sh
runtime/venv/bin/python -B tests/tuning_gpu.py on
runtime/venv/bin/python -B tests/tuning_gpu.py off
```

途中で上限時間に達した場合は `on --resume` で保存済み結果を再利用できます。この再開試験では候補ごとの測定を1ms・1回に制限します。

常駐サービスは `deploy/clef2-worker.service` を元に導入済みです。旧workerと同時接続しない状態で切り替えています。現在の稼働設定は末尾の「常駐切り替え」を参照してください。

## 検証状況

2026-10-15時点で28テストが成功しました。49件の静的判定は旧実装と一致し、49件すべてで判定質問・原文・エンコード後のトークン列も旧実装と一致しました。総入力は1240〜1254トークンで、2048以内です。モデル全15ファイルのハッシュ一致、保存済みTunableOp 55件の読み込み、systemdユニットの構文、秘密ファイルの権限・Git除外も確認しました。

旧worker・旧Web画面・旧GPU APIをすべて停止して実GPU比較を実施しました。24件を各3回処理した72回の判定は全件一致し、旧構成も各入力の3回で判定が揺れませんでした。比較元もHTTPサーバーを起動せず、検証済みの旧起動コードの判定処理を直接呼び出しました。

| 指標 | 旧構成 | clef2 |
| --- | --- | --- |
| 掲載判定中央値 | 1.135秒 | 1.140秒 |
| 95パーセンタイル | 1.145秒 | 1.148秒 |
| 最大値 | 1.155秒 | 1.186秒 |
| 起動後の初回判定 | 4.259秒 | 4.310秒 |

中央値の比は1.004で、110%以内の基準に合格しました。エンジン起動・初回コンパイルの時間は上記の判定時間に含めません。clef2の実GPUとローカルWebSocketを通した49件も成功し、静的25件は推論0回、非静的24件は推論1回、不正入力はreview返信を確認しました。

オンライン調整は専用CSVで55件から85件への保存を確認しました。最初の試験は起動準備の調整に約312秒かかり、判定完了前に試験全体の6分上限で終了しました。保存済み結果から測定上限を1ms・1回に縮めて再開し、85件から98件への追加保存とallow判定を確認しました。この判定は約95.4秒でした。別プロセスで追加調整を無効にして再起動し、CSVハッシュと件数が変わらず再利用され、同じallow判定になることを確認しました（その初回判定は約0.90秒）。後者は1入力の値で、上記72回の性能比較とは別の測定です。運用CSVと原本は変更していません。

実バックエンドでのheldへの遷移と、clef2用systemdサービスでの障害時自動再起動の実運用試験は未実施です。比較結果・調整確認結果は `runtime/comparison.json`、`runtime/tuning-validation.json` に保存しています。

旧3サービスは21:54:47に停止し、22:13:11にすべて復旧しました（JST、停止18分24秒）。旧Clef API・旧Web画面のヘルスチェック200と、旧workerのバックエンドへの再接続を確認しました。自動復旧タイマーは停止済みで、検証用GPUサービスも終了しています。復旧記録と全体の確認結果は `runtime/restoration.json`、`runtime/validation-summary.json` に保存しています。

## 常駐切り替え

2026-10-15 23:24 JST、ユーザーの指示により旧3サービスを停止・自動起動無効にし、`clef2-worker.service` へ切り替えました。clef2は稼働・自動起動有効で、バックエンドへの接続と再起動回数0を確認しました。旧Web画面・旧Clef APIは停止しており、7010・11634番ポートも閉じています。

現在の `.env` の値を維持し、推論上限30秒、待機16件、オンライン調整OFF、保存済み調整結果の利用もOFFで起動しています。設定値は変更していません。切り替え時点では新しい投稿がなく、実バックエンドからのタスクの判定返信はまだ観測していません。

状態確認は `systemctl --user status clef2-worker.service`、ログ確認は `journalctl --user -u clef2-worker.service -f` で行えます。切り替え記録は `runtime/cutover.json`。接続確認後、失敗時のロールバック予約は解除済みです。
