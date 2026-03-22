def main():

    model_path = "src/ceramicraft_ai_secure_agent/model/fraud_logistic_regression.pkl"

    with open(model_path, "rb") as f:
        # 使用 pickle 的调试工具查看指令流
        import pickletools

        try:
            # 打印最后 100 条指令，寻找 GLOBAL 指令
            for op, arg, pos in pickletools.genops(f):
                if "GLOBAL" in op.name:
                    print(f"Pos: {pos} | Op: {op.name} | Arg: {arg}")
        except Exception as e:
            print(f"分析中断: {e}")


if __name__ == "__main__":
    main()
