#!/usr/bin/env python3
"""
将PDF文件转换为PNG格式的脚本
用于在GitHub README中显示图片
"""

import os
import fitz  # PyMuPDF

def convert_pdf_to_png(pdf_path, png_path, dpi=150):
    """将单个PDF文件转换为PNG格式"""
    try:
        # 打开PDF文件
        doc = fitz.open(pdf_path)

        # 获取第一页
        page = doc[0]

        # 设置DPI
        mat = fitz.Matrix(dpi/72, dpi/72)

        # 渲染页面为像素图
        pix = page.get_pixmap(matrix=mat)

        # 保存为PNG文件
        pix.save(png_path)

        # 关闭文档
        doc.close()

        print(f"✓ 成功转换: {pdf_path} -> {png_path}")
        return True

    except Exception as e:
        print(f"✗ 转换失败: {pdf_path} - {e}")
        return False

def main():
    # 设置路径
    figures_dir = "/home/github/V-Sparse/figures"

    # 要转换的PDF文件
    pdf_files = [
        "framework.pdf",
        "motivation.pdf",
        "svsc.pdf",
        "video_similarity.pdf"
    ]

    print("开始转换PDF文件为PNG格式...")
    print("=" * 50)

    success_count = 0
    total_count = len(pdf_files)

    for pdf_file in pdf_files:
        pdf_path = os.path.join(figures_dir, pdf_file)
        png_file = pdf_file.replace(".pdf", ".png")
        png_path = os.path.join(figures_dir, png_file)

        if convert_pdf_to_png(pdf_path, png_path):
            success_count += 1

    print("=" * 50)
    print(f"转换完成: {success_count}/{total_count} 个文件成功")

    if success_count == total_count:
        print("\n所有PDF文件已成功转换为PNG格式！")
        print("现在可以在GitHub README中显示这些图片了。")
    else:
        print("\n部分文件转换失败，请检查错误信息。")

if __name__ == "__main__":
    main()
