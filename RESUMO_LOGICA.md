# Alocação de Aeronaves em Posições (Confins) — resumo da lógica

Este projeto resolve um **problema de alocação de aeronaves em posições (portões/remotos/estacionamento)** ao longo do tempo.
A ideia central é transformar o cronograma de voos em **operações com janelas de tempo** e escolher, para cada operação, **uma posição compatível**, respeitando conflitos e regras operacionais.

O código foi estruturado para ficar bem explicável: primeiro ele **prepara dados e regras**, depois monta um **modelo de otimização (Gurobi)** com variáveis/constraints claras e, por fim, gera **saídas + visualização (Streamlit)**.

---

## 1) Entradas (CSV)

### `voos.csv`
Usado para criar o cronograma base (chegadas/partidas). O pré-processamento espera colunas como:

- `Chegada Partida` (chegada/partida)
- `Data`
- `Horário (Hora Local)`
- `Empresa`
- `Voo`
- `Aeronave` (código do tipo, ex.: 763, 339, 320…)
- `Assentos` (usado como proxy de PAX)
- `Origem Destino`

### `posicoes.csv`
Define as posições físicas disponíveis e suas características:

- `Posicao` (id do portão/posição, ex.: 117)
- `Tipo` (Contato / RCmoto / CstacionamCnto)
- `Patio` (para agrupar posições em áreas)
- `Aeronave` (regra de compatibilidade por categoria, ex.: C, DE)

### `categoriaaeronaves.csv`
Mapeia código de aeronave para categoria operacional (A/C/D/E etc.):

- `Aeronave`
- `Categoria`

### `especificacoesaeronaves.csv` (opcional)
Existe no repositório, mas **não é essencial** para o modelo atual (o solver já funciona sem ela). Pode ser útil para futuras extensões.

---

## 2) Conceitos principais

### Visita (`visit`)
Uma **visita** representa o tempo em que uma aeronave fica no aeroporto: **chegada → partida**.
O código tenta **parear** uma chegada com a próxima partida compatível (mesma empresa e mesmo código de aeronave), dentro de limites configuráveis:

- `min_turnaround_minutes`
- `max_turnaround_minutes`

Se não encontra um pareamento, aquela chegada/partida não entra como visita.

### Operação (`operation`)
Uma visita vira **uma ou três operações**, dependendo do tempo de permanência:

- **Curta permanência** → 1 operação: `turnaround` (Chegada → Partida)
- **Longa permanência** (acima de `tow_threshold_minutes`) → 3 operações:
  - `arrival` (desembarque)
  - `parking` (estacionado)
  - `departure` (embarque)

Isso é uma escolha bem forte de modelagem: deixa o modelo mais realista e também facilita explicar o que está acontecendo no Gantt.

---

## 3) Compatibilidade: quem pode usar qual posição

A compatibilidade é tratada em dois níveis (o que ajuda muito na explicabilidade):

1) **Regra do CSV** (`posicoes.csv`): cada posição tem uma “capacidade” por categoria.
2) **Regras adicionais do aeroporto** (Confins), centralizadas em configuração.

### 3.1 Regra base do `posicoes.csv`
O campo `Aeronave` do `posicoes.csv` é interpretado assim:

- `C`  → a posição aceita **A e C**
- `DE` → a posição aceita **A, D e E** (A é menor e pode usar qualquer posição)

> Importante: além disso, o código trata a **Categoria A como universal**: se a aeronave é A, ela é considerada compatível com qualquer posição.

### 3.2 Regra física adicional (Confins): D/E só em stands específicos
Para refletir a limitação física/operacional do aeroporto, o código impõe:

- Se a aeronave for **categoria D ou E**, então ela **só pode ser alocada** nos portões:
  - **117, 120, 123**

Essa regra fica configurável via `ModelConfig.de_allowed_stands`, o que é bom por dois motivos:

- a regra fica **documentável e auditável** (é fácil apontar onde está)
- caso o aeroporto mude ou você queira testar cenários, basta ajustar a configuração

### 3.3 Estacionamento “somente parking”
Posições do tipo `estacionamento` são marcadas como **parking-only**.
O modelo permite essas posições **apenas** para operações do tipo `parking` (a etapa em que o avião está estacionado sem embarque/desembarque).

---

## 4) Adjacência: “bloqueio” dos portões vizinhos (regra de segurança/folga)

O projeto implementa uma regra operacional importante:

- Se uma aeronave **D/E** está em um portão, então os **dois portões adjacentes** ficam **bloqueados** durante o período em que existir sobreposição de tempo.

### Como o código define “adjacente”
A adjacência é gerada automaticamente a partir do `posicoes.csv`:

- Duas posições são adjacentes se forem **consecutivas** (diferença 1 no número)
- e estiverem no **mesmo pátio** (`Patio`)

Exemplos:

- 120 é adjacente a 119 e 121
- 123 é adjacente a 122 e 124

### Como o bloqueio é aplicado
O bloqueio não é “global”; ele é aplicado de forma correta **no tempo**:

- só existe bloqueio entre operações que **se sobrepõem** (com um buffer de segurança `turnaround_buffer_minutes`)

Na prática, se uma operação D/E ocupa o stand 120 em um intervalo, o modelo força que **nenhuma operação sobreposta** possa usar os stands 119 e 121.

Essa é uma restrição muito bem alinhada com a realidade e ainda mantém o modelo “econômico”: você evita bloquear adjacências quando os horários não conflitam.

---

## 5) Modelo de otimização (Gurobi)

### 5.1 Variáveis de decisão

- **Alocação**:  
  $x_{op,stand} \in \{0,1\}$ indica se a operação `op` foi atribuída ao `stand`.

- **Reboque** (quando há sucessão):  
  $y_{op} \in \{0,1\}$ indica se há reboque entre uma operação e sua sucessora (ex.: `arrival → parking`, `parking → departure`).

### 5.2 Restrições principais

1) **Alocação única por operação**  
Cada operação precisa estar em exatamente uma posição compatível.

2) **Conflito temporal na mesma posição**  
Se duas operações se sobrepõem, elas não podem ocupar o mesmo stand.

3) **Bloqueio de adjacência para D/E**  
Se uma operação D/E está em um stand, então, para operações sobrepostas, os stands adjacentes ficam proibidos.

4) **Definição/forçamento de reboque**  
Se operação e sucessora ficam em stands diferentes, então `y=1` (reboque).  
Além disso, se uma operação pode ir a um stand que a sucessora não pode, o modelo já sabe que isso implica reboque.

### 5.3 Funções objetivo (configuráveis)
O projeto permite alternar o “foco” da solução:

- `walking_distance`: minimizar uma proxy de distância (baseada no tipo do stand + ajustes por pátio/ordem)
- `contact_share`: maximizar PAX alocados em posições de contato
- `tow_count`: minimizar número de reboques
- `revenue`: maximizar uma proxy de receita (fator por tipo de posição)

Isso é ótimo para explicabilidade porque você consegue comparar soluções e dizer: “aqui priorizamos conforto (contato), aqui priorizamos logística (reboque)”.

---

## 6) Saídas

O solver gera:

- `outputs/alocacao_resultado.csv`: uma linha por operação com:
  - `operation_type`, `start_time`, `end_time`, `stand_id`, `stand_type`, `aircraft_category`, `pax`, etc.

- `outputs/reboques_resultado.csv`: lista de reboques (quando `y=1`) com:
  - operação de origem, sucessora e indicador de tow

---

## 7) Visualização e explicabilidade no Streamlit

A interface do Streamlit foi pensada para ajudar leitura por quem não é técnico:

- KPIs (operações, posições usadas, reboques, % PAX em contato, long-stay)
- Gantt por posição com:
  - rótulos “Chegada / Estacionado / Partida”
  - tooltip com **Chegada, Partida e Tempo em solo** por visita
  - duração da etapa (ex.: “45 min”, “2 h 30 min”)
- Lista “**voos por portão**” no dia selecionado, com filtro de portão

A leitura fica intuitiva: verde = Chegada, cinza = Estacionado, vermelho = Partida, azul = Chegada→Partida (visita curta).

---

## 8) Onde as regras “moram” (para auditoria)

Se você precisar justificar/defender o modelo em relatório/apresentação, estes são os blocos conceituais mais importantes:

- Preparação/normalização dos dados (voos, posições, categorias)
- Reconstrução de visitas (pareamento chegada-partida)
- Construção de operações (turnaround vs arrival/parking/departure)
- Compatibilidade categoria × posição (inclui regra A universal + restrição física D/E)
- Conflitos temporais
- Adjacência (bloqueio de vizinhos quando D/E ocupa stand)
- Definição de reboques e objetivos

---

## Observações práticas

- Se a instância ficar **inviável** (status `INFEASIBLE`), normalmente é por falta de capacidade compatível em algum período: as regras de compatibilidade (especialmente D/E + bloqueio de adjacentes) podem “apertar” bastante o problema.
- O conjunto de stands permitidos para D/E é uma boa alavanca de cenário (ex.: adicionar 126, se fizer sentido operacionalmente).
