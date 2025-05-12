import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Metrics from Battery CSV with Annotations"
    )
    parser.add_argument(
        "--csv-path", type=str, required=True,
        help="输入的 CSV 文件路径，需包含 'Cycle_Index', 'Discharge Time (s)', 'Time constant current (s)', \
             'Min. Voltage Charg. (V)', 'Max. Voltage Dischar. (V)' 列"
    )
    parser.add_argument(
        "--out-file", type=str, default="all_cycles.png",
        help="输出的图像文件名 (png)"
    )
    parser.add_argument(
        "--cycles", type=float, nargs="+", default=[100, 300, 500, 700, 900],
        help="要绘制的 Cycle_Index 列表"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sns.set_style('whitegrid')
    palette = sns.color_palette(n_colors=len(args.cycles))

    # 读取数据
    df = pd.read_csv(args.csv_path)

    # 创建 1x3 子图：放电时间-电流、放电时间-充电电压、放电时间-放电电压
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))

    for i, cyc in enumerate(args.cycles):
        sub = df[df['Cycle_Index'] == cyc]
        # 时间常数电流
        ax0.plot(
            sub['Discharge Time (s)'],
            sub['Time constant current (s)'],
            label=f'Cycle {int(cyc)}', color=palette[i]
        )
        # 最小充电电压
        ax1.plot(
            sub['Discharge Time (s)'],
            sub['Min. Voltage Charg. (V)'],
            label=f'Cycle {int(cyc)}', color=palette[i]
        )
        # 最大放电电压
        ax2.plot(
            sub['Discharge Time (s)'],
            sub['Max. Voltage Dischar. (V)'],
            label=f'Cycle {int(cyc)}', color=palette[i]
        )

    # 格式化子图
    ax0.set_title('Time Constant Current vs Discharge Time')
    ax0.set_xlabel('Discharge Time (s)')
    ax0.set_ylabel('Time Constant Current (s)')
    ax0.legend(frameon=True)

    ax1.set_title('Min. Charging Voltage vs Discharge Time')
    ax1.set_xlabel('Discharge Time (s)')
    ax1.set_ylabel('Min. Voltage Charg. (V)')
    ax1.legend(frameon=True)

    ax2.set_title('Max. Discharging Voltage vs Discharge Time')
    ax2.set_xlabel('Discharge Time (s)')
    ax2.set_ylabel('Max. Voltage Dischar. (V)')
    ax2.legend(frameon=True)

    # —— 添加标注 ——
    # 示例：在第一个子图标注 F6 & F7
    ax0.axhline(0.8, linestyle='--', color='k')
    ax0.axvline(100, linestyle='--', color='k')
    ax0.annotate('F6', xy=(0, 0.8), xytext=(0.1*df['Discharge Time (s)'].max(), 0.85*df['Time constant current (s)'].max()),
                  arrowprops=dict(arrowstyle='->', linestyle='--'))
    ax0.axhline(0.4, linestyle='--', color='k')
    ax0.axvline(df['Discharge Time (s)'].max(), linestyle='--', color='k')
    ax0.annotate('F7', xy=(0, 0.4), xytext=(0.1*df['Discharge Time (s)'].max(), 0.45*df['Time constant current (s)'].max()),
                  arrowprops=dict(arrowstyle='->', linestyle='--'))

    # 在第二个子图标注 F4 & F5
    ax1.axhline(3.4, linestyle='--', color='k')
    ax1.annotate('F4', xy=(0, 3.4), xytext=(0.05*df['Discharge Time (s)'].max(), 3.45),
                  arrowprops=dict(arrowstyle='->', linestyle='--'))
    ax1.axhline(4.2, linestyle='--', color='k')
    ax1.axvline(150, linestyle='--', color='k')
    ax1.annotate('F5', xy=(0, 4.2), xytext=(0.1*df['Discharge Time (s)'].max(), 4.25),
                  arrowprops=dict(arrowstyle='->', linestyle='--'))

    # 在第三个子图标注 F1, F2, F3
    ax2.axhline(3.25, linestyle='--', color='k')
    ax2.annotate('F1', xy=(0, 3.25), xytext=(0.1*df['Discharge Time (s)'].max(), 3.3),
                  arrowprops=dict(arrowstyle='->', linestyle='--'))
    ax2.axhline(3.6, linestyle='--', color='k')
    ax2.axvline(75, linestyle='--', color='k')
    ax2.annotate('F2', xy=(75, 3.6), xytext=(80, 3.62),
                  arrowprops=dict(arrowstyle='->', linestyle='--'))
    first = df[df['Cycle_Index']==args.cycles[0]].iloc[0]
    ax2.annotate('F3', xy=(first['Discharge Time (s)'], first['Max. Voltage Dischar. (V)']),
                  xytext=(first['Discharge Time (s)']-20, first['Max. Voltage Dischar. (V)']+0.1),
                  arrowprops=dict(arrowstyle='->', linestyle='--'))

    plt.tight_layout()
    plt.savefig(args.out_file, dpi=300)
    print(f"✔ 图已保存到 {args.out_file}")


if __name__ == '__main__':
    main()
