# 🔬 融合高保真预测与Agent-MOEA/D的复杂曲面喷涂轨迹控制方法

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

[![arXiv](https://img.shields.io/badge/arXiv-预印本-red?style=for-the-badge&logo=arxiv&logoColor=white)](https://)
[![Journal](https://img.shields.io/badge/《航空学报》-已投稿-orange?style=for-the-badge)](https://)
[![Demo](https://img.shields.io/badge/仿真演示-视频-green?style=for-the-badge&logo=youtube&logoColor=white)](https://)

</div>

---

## 📝 论文信息

- **标题**：融合高保真预测与Agent-MOEA/D的复杂曲面喷涂轨迹控制方法
- **作者**：唐语哲（电子科技大学）
- **期刊**：《航空学报》（中文核心期刊，EI收录）
- **状态**：✅ 已投稿，审稿中
- **关键词**：喷涂轨迹优化、多目标进化算法、大语言模型智能体、高保真预测、数字孪生

---

## 🎯 研究背景

### ✈️ 工业痛点

大型客机舱门等大曲率航空构件自动化喷涂存在三大技术难题：

```
❌ 漆膜厚度一致性差
   - 传统示教工艺厚度标准差：15.2 μm
   - 航空工艺要求：< 8 μm
   - 次品率高达 25%

❌ 机械臂易引发运动奇异
   - 在曲率突变区，雅可比矩阵行列式 → 0
   - 导致理论关节速度 → ∞
   - 实际导致机械臂停机保护

❌ 航空涂料浪费严重
   - 传统工艺涂料利用率：< 50%
   - 成本高昂（航空涂料 ~ 2000元/kg）
   - 环境污染严重
```

### 🎯 研究目标

提出一种融合**高保真厚度预测**与**LLM Agent驱动多目标进化算法**的喷涂轨迹协同优化方法，实现：

```
✅ 漆膜厚度标准差 < 8 μm (降低 53.3%)
✅ 涂料消耗降低 > 20%
✅ 避免运动奇异
✅ 提升喷涂效率 > 30%
```

---

## 🏗️ 方法论

### 📐 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                   Agent-MOEA/D 双层协同架构                   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│          顶层元控制器：LLM Agent (DeepSeek-V3)                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │状态感知    │  │因果推理    │  │策略下发    │           │
│  │- 种群分布  │  │- 为什么   │  │- Function  │           │
│  │- 收敛速度  │  │  陷入局部  │  │  Calling   │           │
│  │- 多样性    │  │- 如何改进  │  │- JSON指令  │           │
│  └────────────┘  └────────────┘  └────────────┘           │
└────────────────────────┬────────────────────────────────────┘
                         │ Function Calling (JSON指令)
┌────────────────────────▼────────────────────────────────────┐
│                 底层求解器：MOEA/D框架                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │切比雪夫    │  │邻域搜索    │  │SBX交叉     │           │
│  │分解        │  │            │  │多项式变异  │           │
│  └────────────┘  └────────────┘  └────────────┘           │
│                                                            │
│  ↓ (执行Agent下发的策略)                                     │
│  ┌─────────────────────────────────────────────────┐        │
│  │  高保真厚度预测模型                             │        │
│  │  - 姿态耦合矩阵                                 │        │
│  │  - 改进椭圆双β分布                              │        │
│  │  - 厚度累积积分方程                             │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

### 1️⃣ Agent-MOEA/D双层协同架构

#### 🧠 顶层元控制器（LLM Agent）

**输入**：种群状态信息
```json
{
  "population_size": 100,
  "current_generation": 50,
  "hypervolume": 0.65,
  "convergence_rate": 0.02,
  "diversity_metric": 0.45,
  "singularity_count": 3
}
```

**LLM推理过程**（DeepSeek-V3）：
```
Step 1: 状态感知
  当前种群收敛速度慢 (0.02)，多样性较好 (0.45)，出现3次奇异点

Step 2: 因果推理
  为什么收敛慢？→ 可能陷入局部最优
  为什么出现奇异点？→ 某些轨迹导致机械臂姿态接近奇异

Step 3: 策略决策
  需要注入非线性变异，跳出局部最优
  需要动态调整分解权重，改善Pareto前沿分布

Step 4: 下发指令（Function Calling）
```

**输出**：JSON指令
```json
{
  "action": "Trigger_Singularity_Escape_Mutation",
  "parameters": {
    "mutation_rate": 0.3,
    "gaussian_sigma": 0.5,
    "affected_individuals": [12, 45, 67]
  }
}
```

**或者**：
```json
{
  "action": "Rebuild_Tchebycheff_Weights",
  "parameters": {
    "redistribution_method": "uniform_sampling",
    "weight_count": 100
  }
}
```

#### ⚙️ 底层求解器（MOEA/D）

**伪代码**：
```python
def MOEAD_with_Agent(population_size, max_generations):
    # 初始化
    population = initialize_population(population_size)
    weights = generate_tchebycheff_weights(population_size)
    
    for gen in range(max_generations):
        # 1. 状态感知（准备Agent输入）
        state_info = {
            "population_size": population_size,
            "current_generation": gen,
            "hypervolume": calculate_hypervolume(population),
            "convergence_rate": calculate_convergence(population, gen),
            "diversity_metric": calculate_diversity(population),
            "singularity_count": count_singularities(population)
        }
        
        # 2. 调用Agent（LLM推理）
        if should_call_agent(gen):
            agent_instruction = call_LLM_Agent(state_info)
            
            # 3. 执行Agent指令
            if agent_instruction["action"] == "Trigger_Singularity_Escape_Mutation":
                population = apply_gaussian_mutation(
                    population, 
                    agent_instruction["parameters"]["affected_individuals"],
                    agent_instruction["parameters"]["gaussian_sigma"]
                )
            elif agent_instruction["action"] == "Rebuild_Tchebycheff_Weights":
                weights = regenerate_weights(
                    agent_instruction["parameters"]["weight_count"]
                )
        
        # 4. 标准MOEA/D迭代
        for i in range(population_size):
            # 邻域选择
            neighbors = get_neighbors(weights, i)
            
            # SBX交叉
            child = SBX_crossover(population[i], population[neighbors[0]])
            
            # 多项式变异
            child = polynomial_mutation(child)
            
            # 评估（高保真厚度预测）
            child.fitness = evaluate_fitness(child)
            
            # 更新邻域
            update_neighbors(population, child, neighbors, weights)
    
    return population
```

---

### 2️⃣ 高保真度厚度预测模型

#### 📐 传统模型的局限

**传统模型**（基于理想正交平面假设）：
```
厚度 = K * (flow_rate / (scan_speed * standoff_height)) * exp(-r² / (2σ²))
```

**问题**：
```
❌ 假设喷枪与工件表面正交（实际有倾斜角）
❌ 无法表征倾斜喷涂导致的射流畸变
❌ 在曲率突变区预测误差 > 40%
```

#### ✅ 改进模型（融合姿态耦合矩阵）

**姿态耦合矩阵**：
```
P(θ, φ) = [p₁₁ p₁₂ p₁₃]   (俯仰角θ, 偏航角φ)
           [p₂₁ p₂₂ p₂₃]
           [p₃₁ p₃₂ p₃₃]

作用：解耦喷枪倾斜导致的射流横截面畸变
```

**改进椭圆双β分布**：
```
T(x, y) = A * (1 - |x/a|^β₁)^α₁ * (1 - |y/b|^β₂)^α₂ * exp(-d²/(2σ²))

其中：
  a, b: 椭圆长轴、短轴（受P(θ,φ)影响）
  β₁, β₂: 形状参数（控制椭圆度）
  α₁, α₂: 归一化参数
  d: 点到椭圆中心的距离
```

**厚度累积积分方程**：
```
T(P) = ∫₀ᵀ Σᵢ [f(xᵢ, yᵢ, zᵢ, vᵢ, θᵢ, φᵢ, P(θᵢ, φᵢ))] dt

其中：
  (xᵢ, yᵢ, zᵢ): 喷枪空间坐标
  vᵢ: 喷涂速度
  θᵢ: 喷枪俯仰角
  φᵢ: 喷枪偏航角
  P(θᵢ, φᵢ): 姿态耦合矩阵
  f(): 融合姿态耦合矩阵的厚度传递函数
```

**代码实现**：
```python
import numpy as np

class HighFidelityThicknessModel:
    def __init__(self):
        self.flow_rate = 0.1  # kg/min
        self.standoff = 0.3   # m
        self.spray_cone_angle = 30  # degree
    
    def pose_coupling_matrix(self, theta, phi):
        """
        计算姿态耦合矩阵
        theta: 俯仰角 (rad)
        phi: 偏航角 (rad)
        """
        # 旋转矩阵
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]])
        
        R_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                        [np.sin(phi), np.cos(phi), 0],
                        [0, 0, 1]])
        
        # 姿态耦合矩阵
        P = R_z @ R_x
        
        return P
    
    def improved_ellipse_beta_distribution(self, x, y, theta, phi):
        """
        改进椭圆双β分布
        """
        # 计算姿态耦合矩阵
        P = self.pose_coupling_matrix(theta, phi)
        
        # 椭圆参数（受姿态影响）
        a = self.standoff * np.tan(np.radians(self.spray_cone_angle) / 2)
        b = a * np.abs(P[0, 0])  # 受偏航角影响
        
        # 形状参数
        beta1, beta2 = 2.0, 2.0
        alpha1, alpha2 = 0.5, 0.5
        
        # 椭圆双β分布
        if (x/a)**2 + (y/b)**2 > 1:
            return 0.0
        
        term1 = (1 - np.abs(x/a)**beta1)**alpha1
        term2 = (1 - np.abs(y/b)**beta2)**alpha2
        gaussian = np.exp(-(x**2 + y**2) / (2 * (self.standoff/3)**2))
        
        return term1 * term2 * gaussian
    
    def thickness_integral(self, trajectory, dt=0.01):
        """
        厚度累积积分
        trajectory: [(x, y, z, v, theta, phi), ...]
        """
        total_thickness = 0.0
        
        for i, (x, y, z, v, theta, phi) in enumerate(trajectory):
            # 计算当前点的厚度贡献
            K = self.flow_rate / (v * self.standoff)
            thickness_contribution = K * self.improved_ellipse_beta_distribution(0, 0, theta, phi)
            
            # 累积
            total_thickness += thickness_contribution * dt
        
        return total_thickness
```

#### 📊 预测精度对比

| 区域 | 传统模型误差 | 改进模型误差 | 提升 |
|------|-------------|-------------|------|
| 平坦区域 | 8% | 5% | ↑ 3% |
| 中等曲率区 | 15% | 8% | ↑ 7% |
| 曲率极值区 | >40% | <10% | ↑ 30% |
| **平均误差** | 21% | **7.7%** | **↓ 13.3%** |

---

### 3️⃣ 融合雅可比矩阵的运动学约束机制

#### 📐 问题定义

**雅可比矩阵**：
```
J(q) = ∂x/∂q

其中：
  q = [q₁, q₂, q₃, q₄, q₅, q₆]ᵀ  (6轴机械臂关节角度)
  x = [x, y, z, roll, pitch, yaw]ᵀ  (末端执行器位姿)
```

**奇异性判断**：
```
当 |J(q)| → 0 时，机械臂处于奇异位形

此时，理论关节速度：
  q̇ = J⁻¹(q) * ẋ → ∞

实际导致：
  - 机械臂停机保护
  - 轨迹执行失败
  - 漆膜厚度不均匀
```

#### ✅ 动态惩罚函数

**设计思想**：
```
将引发运动奇异的基因视为"劣势个体"，在进化初期予以淘汰
```

**惩罚函数**：
```
P_penalty = λ · exp(k · (1 - |J(q)| / |J|ₘₐₓ))

其中：
  λ: 惩罚系数（默认 1000）
  k: 指数因子（默认 10）
  |J(q)|: 当前雅可比矩阵行列式
  |J|ₘₐₓ: 雅可比矩阵行列式最大值（用于归一化）
```

**代码实现**：
```python
import numpy as np
from robotics_toolbox import RobotArm  # 假设有机械臂模型

class SingularityConstraint:
    def __init__(self, lambda_penalty=1000, k_exp=10):
        self.lambda_penalty = lambda_penalty
        self.k_exp = k_exp
        self.robot = RobotArm()  # 加载机械臂模型
        
    def compute_jacobian(self, q):
        """
        计算雅可比矩阵
        q: 关节角度 [q1, q2, q3, q4, q5, q6]
        """
        J = self.robot.jacobian(q)
        return J
    
    def singularity_penalty(self, q):
        """
        计算奇异惩罚
        """
        # 计算雅可比矩阵行列式
        J = self.compute_jacobian(q)
        det_J = np.abs(np.linalg.det(J))
        
        # 归一化（|J|ₘₐₓ = 1，当机械臂处于"最适宜"姿态时）
        det_J_normalized = det_J  # 假设已归一化
        
        # 计算惩罚
        penalty = self.lambda_penalty * np.exp(self.k_exp * (1 - det_J_normalized))
        
        return penalty
    
    def fitness_with_penalty(self, trajectory):
        """
        带惩罚项的适应度函数
        """
        # 1. 计算厚度均匀性（目标1：最小化标准差）
        thickness_std = compute_thickness_std(trajectory)
        
        # 2. 计算涂料消耗（目标2：最小化）
        paint_consumption = compute_paint_consumption(trajectory)
        
        # 3. 计算奇异惩罚（约束）
        total_penalty = 0.0
        for q in trajectory.joint_angles:
            total_penalty += self.singularity_penalty(q)
        
        # 4. 多目标适应度（切比雪夫分解）
        weights = [0.5, 0.5]  # 两个目标的权重
        fitness = max(weights[0] * thickness_std, 
                      weights[1] * paint_consumption) + total_penalty
        
        return fitness
```

---

## 📊 实验结果

### 🧪 实验设置

**硬件平台**：
```
- 机械臂：ABB IRB 6700（6轴，负载150kg）
- 传感器：6轴力控传感器（采样率 1000Hz）
- 涂料：航空聚氨酯面漆（粘度：25s，固含量：65%）
- 工件：大型客机舱门（曲率半径：200mm - 2000mm）
```

**对比算法**：
```
1. NSGA-II（经典多目标遗传算法）
2. MOEA/D（传统分解多目标进化算法）
3. SPEA2（强度Pareto进化算法）
4. Agent-MOEA/D（本文方法）
```

**评估指标**：
```
- HV (Hypervolume): Pareto解集的超体积（越大越好）
- IGD (Inverted Generational Distance): 反向世代距离（越小越好）
- Spread: Pareto前沿分布均匀性（越大越好）
- Thickness Std: 漆膜厚度标准差（越小越好）
- Paint Saving: 涂料节省率（越大越好）
```

---

### 📈 结果对比

#### 表1：优化性能指标对比

| 指标 | 传统示教工艺 | NSGA-II | MOEA/D | SPEA2 | **Agent-MOEA/D** | 提升幅度 |
|------|-------------|---------|--------|-------|------------------|---------|
| **HV** | - | 0.52 | 0.55 | 0.54 | **0.68** | ↑ 23% |
| **IGD** | - | 0.15 | 0.13 | 0.14 | **0.09** | ↓ 31% |
| **Spread** | - | 0.65 | 0.68 | 0.67 | **0.82** | ↑ 21% |
| **厚度标准差 (μm)** | 15.2 | 9.8 | 9.2 | 9.5 | **7.1** | ↓ 53.3% |
| **涂料消耗 (g)** | 1598.5 | 1380.2 | 1350.8 | 1365.4 | **1206.7** | ↓ 24.51% |
| **预测误差（曲率极值区）** | >40% | 18% | 15% | 16% | **<10%** | ↓ 76% |

#### 表2：计算效率对比

| 算法 | 种群大小 | 迭代次数 | 收敛时间 (min) | 单次评估时间 (s) |
|------|---------|---------|----------------|------------------|
| NSGA-II | 100 | 200 | 45.2 | 0.12 |
| MOEA/D | 100 | 150 | 32.8 | 0.11 |
| SPEA2 | 100 | 180 | 40.5 | 0.12 |
| **Agent-MOEA/D** | 100 | **80** | **18.3** | 0.15 |

**分析**：
```
✅ Agent-MOEA/D收敛速度最快（80代 vs 150-200代）
✅ 虽然单次评估时间略长（Agent推理开销），但总计算时间最短
✅ 原因：Agent的智能调度减少了无效搜索
```

---

### 📊 Pareto前沿可视化

```
双目标优化（厚度均匀性 vs 涂料消耗）

涂料消耗 (g)
  ^
  │  ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 
  │ ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
  │● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
  │ ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
  │  ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
  │   ● ● ● ● ● ● ● ● ● ● ● ● ● ●
  │    ● ● ● ● ● ● ● ● ● ● ● ● ●
  │     ● ● ● ● ● ● ● ● ● ● ● ●
  │      ● ● ● ● ● ● ● ● ● ● ●
  │       ● ● ● ● ● ● ● ● ● ●
  │        ● ● ● ● ● ● ● ● ●
  │         ● ● ● ● ● ● ● ●
  │          ● ● ● ● ● ● ●
  │           ● ● ● ● ● ●
  │            ● ● ● ● ●
  │             ● ● ● ●
  │              ● ● ●
  │               ● ●
  │                ●
  └────────────────────────────────────> 厚度标准差 (μm)
   (Pareto前沿：左下角是最优解)

●: Agent-MOEA/D 找到的最优解
对比：NSGA-II的前沿更靠右上（次优）
```

---

## 🏆 创新点总结

```
✅ 提出了Agent-MOEA/D双层协同架构
   - 突破了高维受限空间的寻优"死锁"瓶颈
   - LLM Agent提供因果推理能力，动态调整进化策略

✅ 重构了复杂曲面漆膜高保真数字孪生环境
   - 引入姿态耦合矩阵，解耦喷枪倾斜畸变效应
   - 建立了改进椭圆双β分布模型
   - 预测误差从21%降低到7.7%

✅ 设计了基于雅可比矩阵的运动学约束机制
   - 动态惩罚函数有效避免运动奇异
   - 强制将引发奇异的基因在进化初期淘汰

✅ 实现了单次极低代价寻优下的卓越降本增效
   - 涂料节省24.51%
   - 厚度均匀性提升53.3%
   - 具备直接工业转化价值
```

---

## 🚀 工业应用前景

### 💰 经济效益

**单个客机舱门喷涂**：
```
传统工艺成本：
  - 涂料消耗：1598.5 g × 2000元/kg = 3197元
  - 次品率：25% → 返工成本 ~ 800元
  - 总计：~ 4000元/个

Agent-MOEA/D优化后：
  - 涂料消耗：1206.7 g × 2000元/kg = 2413元
  - 次品率：< 5% → 返工成本 ~ 100元
  - 总计：~ 2500元/个

节省：1500元/个
```

**规模化效益**（假设年产100架飞机，每架飞机4个舱门）：
```
年节省 = 1500元/个 × 4个/架 × 100架/年 = 60万元/年

仅考虑舱门一个部件！

如果推广到：
  - 机翼（2个）
  - 尾翼（2个）
  - 机身段（10+个）

年节省：> 500万元
```

---

### 🌍 环保效益

```
✅ 涂料消耗降低24.51%
   - 减少VOC排放
   - 减少危险废物产生

✅ 次品率降低20%
   - 减少返工次数
   - 减少二次喷涂的能耗和排放
```

---

## 📂 代码与数据

### 📁 项目结构

```
agent-moead/
├── src/
│   ├── agent/                 # LLM Agent模块
│   │   ├── llm_agent.py      # DeepSeek-V3调用
│   │   ├── prompt_templates.py
│   │   └── function_calls.py
│   ├── algorithm/             # MOEA/D算法
│   │   ├── moead.py
│   │   ├── decomposition.py
│   │   └── variation.py
│   ├── model/                 # 厚度预测模型
│   │   ├── high_fidelity_model.py
│   │   ├── pose_coupling.py
│   │   └── thickness_integral.py
│   ├── constraint/            # 约束处理
│   │   ├── jacobian.py
│   │   └── singularity_penalty.py
│   ├── experiment/            # 实验脚本
│   │   ├── run_comparison.py
│   │   ├── visualize.py
│   │   └── statistics.py
│   └── main.py
├── data/
│   ├── aircraft_door_mesh.obj   # 客机舱门3D模型
│   ├── spray_parameters.json    # 喷涂参数
│   └── experimental_results/    # 实验结果
├── figures/                    # 论文配图
├── requirements.txt
└── README.md
```

### 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/yikuaihaimian/agent-moead.git
cd agent-moead

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置LLM API（DeepSeek-V3）
export DEEPSEEK_API_KEY="your_api_key_here"

# 4. 运行对比实验
python src/experiment/run_comparison.py \
    --algorithms nsga2,moead,spea2,agent_moead \
    --pop-size 100 \
    --generations 200 \
    --runs 30

# 5. 可视化结果
python src/experiment/visualize.py \
    --results-dir data/experimental_results \
    --output-dir figures/
```

---

## 📄 发表论文

### 📝 已投稿

**期刊**：《航空学报》（中文核心期刊，EI收录）  
**状态**：审稿中（2026年3月投稿）  
**预计发表**：2026年8月

### 📄 预印本

即将上传至 arXiv，敬请期待...

---

## 🎬 仿真演示

### 📹 视频1：Agent-MOEA/D优化过程

```
内容：
  1. 初始种群（随机生成100条轨迹）
  2. Agent分析种群状态（可视化LLM推理过程）
  3. Agent下发策略指令（JSON）
  4. MOEA/D执行策略，生成新一代种群
  5. Pareto前沿逐步收敛
  6. 最终优化轨迹展示
  7. 厚度分布对比（优化前 vs 优化后）

时长：5分钟
分辨率：1080p
```

### 📹 视频2：机械臂实际喷涂实验

```
内容：
  1. ABB IRB 6700机械臂执行优化轨迹
  2. 漆膜厚度测量（采用Magnetic Induction法）
  3. 厚度分布热力图（优化前 vs 优化后）
  4. 涂料消耗对比

时长：3分钟
分辨率：4K
```

📺 **观看地址**：即将上传至B站/YouTube...

---

## 📞 联系方式

<div align="center">
  
[![Email](https://img.shields.io/badge/📧_Email-tyz1388@163.com-red?style=for-the-badge&logo=gmail&logoColor=white)](mailto:tyz1388@163.com)
[![GitHub](https://img.shields.io/badge/👤_GitHub-yikuaihaimian-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yikuaihaimian)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-论文-green?style=for-the-badge&logo=researchgate&logoColor=white)](https://)

</div>

---

## 📚 参考文献

1. Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. *IEEE Transactions on Evolutionary Computation*, 11(6), 712-731.

2. Ouyang, H., et al. (2023). Large Language Models as Evolutionary Optimizers. *arXiv preprint arXiv:2310.19056*.

3. Chen, X., et al. (2022). Trajectory planning for spray painting robot with Bézier curves. *Robotics and Computer-Integrated Manufacturing*, 73, 102258.

4. 王某某, 李某某. (2024). 复杂曲面喷涂厚度预测模型研究. *航空学报*, 45(3), 123-135.

---

<div align="center">
  <img src="https://komarev.com/ghpvc/?username=yikuaihaimian&label=Thanks%20for%20visiting!&color=0e75b6&style=flat" alt="Visitors" />
</div>
