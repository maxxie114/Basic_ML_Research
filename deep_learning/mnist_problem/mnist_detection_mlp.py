# -*- coding: utf-8 -*-
"""
一个简单的手写数字识别小项目（MLP 版本，且导出的图片标签不重复）：

1. 使用本地的 `mnist_train_small.csv`（28x28 像素灰度图按行展开）
2. 使用 MLPClassifier（多层感知机）进行数字分类
3. 在测试集上绘制混淆矩阵评估分类效果
4. 从测试集中选取若干张图片，根据“预测结果.png”命名导出
5. ✅ 新逻辑：只导出“预测结果标签互不相同”的图片，比如 1.png、3.png、4.png
6. 导出的 PNG 会被放大到 128x128 像素，更容易看清数字

需要安装的库（在终端/命令行里运行）：
    pip install scikit-learn matplotlib numpy pillow
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier  # 使用 MLP 神经网络
from PIL import Image  # 用来把 8x8 图放大并保存为 PNG


def main():
    # 1. 加载数据集
    print("📥 正在从 mnist_train_small.csv 加载手写数字数据集...")
    csv_path = Path(__file__).with_name("mnist_train_small.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"未找到 {csv_path}，请确认数据文件存在。"
        )

    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

    # 前一列是标签（数字 0~9），其余 784 列是 28x28 灰度图展平后的像素
    y = data[:, 0].astype(int)
    X = data[:, 1:]
    images = X.reshape(-1, 28, 28)

    print(f"✅ 数据加载完成，一共包含 {len(X)} 个样本。")

    # 2. 划分训练集和测试集（同时划分 images，保证索引对应）
    print("🔀 正在划分训练集和测试集...")
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        X,
        y,
        images,
        test_size=0.3,
        random_state=42,
        stratify=y,  # 按类别分层抽样，保证比例大致一致
    )
    print(f"✅ 训练集大小：{len(X_train)}，测试集大小：{len(X_test)}")

    # 3. 特征标准化（对 MLP 很重要）
    print("📏 正在对特征进行标准化处理...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ 标准化完成。")

    # 4. 通过网格搜索寻找更优的超参数
    base_clf = MLPClassifier(max_iter=200, random_state=42)
    param_grid = {
        "hidden_layer_sizes": [(128,), (128, 64), (256, 128)],
        "alpha": [1e-3, 1e-2],
        "learning_rate_init": [1e-3, 5e-4],
    }
    search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    print("🔎 正在进行 GridSearchCV 超参数搜索...")
    search.fit(X_train_scaled, y_train)
    clf = search.best_estimator_
    print(
        "✅ 超参数搜索完成：",
        search.best_params_,
        f"（均值验证得分：{search.best_score_:.4f}）",
    )

    print("🤖 使用最佳超参数重新训练最终模型...")
    clf.fit(X_train_scaled, y_train)
    print("✅ 最终模型训练完成。")

    # 额外保存一次损失曲线，方便观察模型收敛情况
    if getattr(clf, "loss_curve_", None):
        print("📉 正在保存 loss 曲线图 loss_curve.png ...")
        plt.figure(figsize=(8, 4))
        plt.plot(
            range(1, len(clf.loss_curve_) + 1),
            clf.loss_curve_,
            marker="o",
            linewidth=1.5,
            markersize=4,
        )
        plt.title("MLP Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("loss_curve.png", dpi=150)
        plt.close()
        print("✅ loss 曲线图已保存为 loss_curve.png。")

    # 5. 在测试集上评估模型表现
    print("📊 正在评估模型在测试集上的表现...")
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 模型在测试集上的准确率为：{acc:.2f}")

    # 6. 生成并保存混淆矩阵
    print("🧮 正在生成混淆矩阵可视化...")
    cm = confusion_matrix(y_test, y_pred, labels=range(10))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Digits classification with MLP (Confusion Matrix)")
    fig.tight_layout()

    confusion_path = "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=200)
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
        plt.close(fig)
        print(
            f"✅ 混淆矩阵已显示，并保存为 {confusion_path}。"
        )
    else:
        plt.close(fig)
        print(
            f"✅ 混淆矩阵已保存为 {confusion_path}（当前后端：{backend}）。"
        )

    # 7. 从测试集中选取图片，放大后保存为“预测结果.png”
    #    ✅ 新逻辑：只保存“预测结果不重复”的图片
    print("💾 正在从测试集中选取图片并保存为放大后的 PNG 文件（预测结果互不重复）...")
    num_to_save = 5         # 希望保存的“不同数字”的数量
    scale = 16               # 放大倍数：8 * 16 = 128，所以输出 128x128 像素

    saved_labels = set()     # 已经保存过的数字标签
    saved_count = 0

    rng = np.random.default_rng()
    shuffled_indices = rng.permutation(len(img_test))

    for idx in shuffled_indices:
        img = img_test[idx]
        pred_label = y_pred[idx]
        # 如果这个数字已经保存过了，就跳过
        if pred_label in saved_labels:
            continue

        # CSV 中的像素范围是 0~255，这里保持同样的范围
        max_val = img.max()
        if max_val == 0:
            img_norm = img
        else:
            img_norm = img / max_val * 255.0

        img_uint8 = img_norm.astype(np.uint8)  # Pillow 需要 uint8 格式

        # 生成放大后的图像
        pil_img = Image.fromarray(img_uint8, mode="L")  # "L" 表示灰度图
        new_size = (img_uint8.shape[1] * scale, img_uint8.shape[0] * scale)
        big_img = pil_img.resize(new_size, Image.NEAREST)  # NEAREST 保留像素块风格

        filename = f"{pred_label}.png"

        big_img.save(filename)

        saved_labels.add(pred_label)
        saved_count += 1

        print(
            f"✅ 已保存第 {saved_count} 张图片，对应的预测结果为 {pred_label}，"
            f"文件名：{filename}，尺寸：{new_size[0]}x{new_size[1]} 像素"
        )

        # 如果已经保存够了，就停止
        if saved_count >= num_to_save:
            break

    if saved_count < num_to_save:
        print(
            f"⚠ 说明：测试集中模型预测到的不同数字一共只有 {saved_count} 种，"
            f"少于期望的 {num_to_save} 种，因此只保存了 {saved_count} 张图片。"
        )

    print("🎉 所有步骤执行完毕，可以在当前项目目录下看到生成的 PNG 图片。")


if __name__ == "__main__":
    main()
