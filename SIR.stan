data {
  int n_sample;            // サンプルサイズ
  int I_obs[n_sample];         // 観測値
  int R_obs[n_sample];
  int S0;
  int R_lag;               // 感染者が回復までにかかる時間
}

parameters {
  real<lower=0> beta[n_sample];         // 感染力
  real<lower=0> gamma;         // 回復力
  real<lower=0> I_sta_zero;            // I_状態の初期値
  real<lower=0> I_sta[n_sample];       // I_状態の推定値
  real<lower=0> R_sta_zero;
  real<lower=0> R_sta[n_sample];
  real<lower=0> s_I;       // 過程誤差の分散
  real<lower=0> s_R;
}

transformed parameters{
  real<lower=0> R_param[n_sample];
  for(i in 1:n_sample){
    R_param[i]=beta[i]*(1-I_sta[i]/S0-R_sta[i]/S0)/gamma;
  }
}

model {
  // 状態の初期値から最初の時点の状態が得られる
  I_sta[1] ~ normal(I_sta_zero, sqrt(s_I));  
  R_sta[1] ~ normal(R_sta_zero, sqrt(s_R));

  // 状態方程式に従い、状態が遷移する

  for (i in 2:R_lag){
    I_sta[i] ~ normal(I_sta[i-1] + beta[i-1]/S0*I_sta[i-1]*(S0-I_sta[i-1]-R_sta[i-1])
                                 - gamma*I_sta[1], sqrt(s_I));
    R_sta[i] ~ normal(R_sta[i-1] + gamma*I_sta[1], sqrt(s_R));
  }
  for (i in R_lag+1:n_sample){
    I_sta[i] ~ normal(I_sta[i-1] + beta[i-1]/S0*I_sta[i-1]*(S0-I_sta[i-1]-R_sta[i-1])
                                 - gamma*I_sta[i-R_lag], sqrt(s_I));
    R_sta[i] ~ normal(R_sta[i-1] + gamma*I_sta[i-R_lag], sqrt(s_R));
  }
  
  //  ポアソン分布に従って観測値が得られる
  for(i in 1:n_sample){
    I_obs[i] ~ poisson(I_sta[i]);
    R_obs[i] ~ poisson(R_sta[i]);
  }
  // 弱情報事前分布。
  beta[1] ~ lognormal(log(0.5),1);
  for(i in 2:n_sample){
     beta[i] ~ lognormal(log(beta[i-1]),0.5);
  }
  gamma ~ lognormal(log(0.5),1);    
  s_I ~ lognormal(log(0.5),1);
  s_R ~ lognormal(log(0.5),1);
}


