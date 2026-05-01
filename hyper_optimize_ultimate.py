import sqlite3, json, sys, argparse, time, os
from collections import Counter
import optuna

ZODIAC_MAP = { ... }  # 保持完整

# ... 所有函数保持不变，只修改 main() ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='newmacau_marksix.db')
    parser.add_argument('--trials', type=int, default=200)   # 每次运行的试验数
    args = parser.parse_args()

    conn = connect_db(args.db)
    issues = load_issues(conn, recent=300)
    conn.close()
    if len(issues) < 80:
        sys.exit(1)

    # 固定 study 名称，存储到文件实现续传
    study = optuna.create_study(
        direction='maximize',
        study_name='ultimate_optimizer',
        storage='sqlite:///optuna_study.db',   # 进度文件
        load_if_exists=True,
    )
    study.optimize(lambda t: objective(t, issues), n_trials=args.trials, show_progress_bar=True)

    best_p = study.best_params
    score, r1, r2, r4, ms1, ms2, ms4 = evaluate(issues, best_p)
    print(f"当前最佳: 一生肖={r1:.3f}(连空{ms1}) 二肖={r2:.3f}(连空{ms2}) 四肖={r4:.3f}(连空{ms4})")

    # 保存最佳参数，供主脚本使用
    with open("best_params_zodiac.json", "w") as f:
        json.dump(best_p, f, indent=2)
    if score > 0:
        print("🎉 已达标！")
    else:
        print("未达标，已保存参数，下次继续优化。")