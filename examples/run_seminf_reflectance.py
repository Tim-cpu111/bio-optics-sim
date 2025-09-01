from biopoptics.models.materials import TissueOpticalProps
from biopoptics.mc.kernels_cpu import simulate_seminf_uniform

if __name__ == "__main__":
    # 组织参数 (典型皮肤)
    props = TissueOpticalProps(mu_a=0.1, mu_s=10.0, g=0.9, n=1.37)

    # TODO: 调用模拟
    # R_d, A = simulate_seminf_uniform(N=10000, props=props)

    print("模拟未实现，待补充")
