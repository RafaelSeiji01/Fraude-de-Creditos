import streamlit as st

class CurriculoPage:
    def __init__(self):
        self.nome = "Rafael Seiji"
        self.cargo = "Desenvolvedor de Software | Full Stack | DevOps"
        self.localizacao = "São Paulo, SP - Brasil"
        self.formacao = "Graduando em Engenharia de Software — FIAP (4º Semestre)"
        self.linkedin = "https://www.linkedin.com/in/rafael-seiji-39961b333/"
        self.github = "https://github.com"
        
    def aplicar_estilo(self):
        st.set_page_config(
            page_title=f"Currículo Profissional | {self.nome}",
            layout="wide"
        )
        st.markdown("""
            <style>
                .stApp {
                    background-color: #f8f9fa;
                }
                .profile-card {
                    background-color: #ffffff;
                    border-radius: 8px;
                    padding: 28px 32px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
                    border: 1px solid #e2e8f0;
                    margin-bottom: 20px;
                }
                .badge {
                    display: inline-block;
                    padding: 4px 10px;
                    font-size: 0.75rem;
                    font-weight: 600;
                    border-radius: 4px;
                    margin-right: 6px;
                    margin-bottom: 6px;
                    letter-spacing: 0.3px;
                }
                .badge-primary { background-color: #e0f2fe; color: #0369a1; }
                .badge-success { background-color: #dcfce7; color: #15803d; }
                .badge-dark { background-color: #f1f5f9; color: #334155; }
                .badge-highlight { background-color: #fef3c7; color: #92400e; }
                .section-title {
                    font-size: 1.05rem;
                    font-weight: 700;
                    color: #0f172a;
                    border-bottom: 1px solid #e2e8f0;
                    padding-bottom: 6px;
                    margin-top: 18px;
                    margin-bottom: 12px;
                }
            </style>
        """, unsafe_allow_html=True)

    def renderizar_cabecalho(self):
        st.title("Perfil Profissional & Portfólio Técnico")
        st.caption("Engenharia de Software • Desenvolvimento Full Stack • Infraestrutura e DevOps")

    def renderizar_conteudo(self):
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        col_esq, col_dir = st.columns([4, 8])
        
        with col_esq:
            st.markdown(f"### {self.nome}")
            st.markdown(f"**{self.cargo}**")
            st.caption(f"{self.localizacao}\n\n{self.formacao}")
            
            st.markdown("""
            <span class="badge badge-highlight">1º Lugar — FIAP Hackathon 2026</span>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">Competências Técnicas</div>', unsafe_allow_html=True)
            st.markdown("""
            <span class="badge badge-primary">Java (Spring Boot)</span>
            <span class="badge badge-primary">Python</span>
            <span class="badge badge-primary">Node.js</span>
            <span class="badge badge-primary">JavaScript</span>
            <span class="badge badge-success">React.js</span>
            <span class="badge badge-success">Tailwind CSS</span>
            <span class="badge badge-success">HTML5 / CSS3</span>
            <span class="badge badge-dark">Redes (Cisco)</span>
            <span class="badge badge-dark">Linux & Shell</span>
            <span class="badge badge-dark">Conceitos Cloud (AWS / Azure)</span>
            <span class="badge badge-dark">Git & GitHub</span>
            <span class="badge badge-dark">Modelagem Relacional (SQL)</span>
            <span class="badge badge-dark">Domain-Driven Design (DDD)</span>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">Informações de Contato</div>', unsafe_allow_html=True)
            st.markdown(f"""
            * **LinkedIn:** [Perfil Profissional]({self.linkedin})
            * **GitHub:** [Repositórios de Código]({self.github})
            * **Disponibilidade:** Estágio / Posições Júnior (Presencial / Híbrido em São Paulo)
            """)

        with col_dir:
            st.markdown('<div class="section-title">Resumo Profissional</div>', unsafe_allow_html=True)
            st.markdown("""
            Desenvolvedor de Software com sólida base em infraestrutura de TI, redes e ciclo de desenvolvimento de software. Experiência na construção de aplicações backend e fullstack, monitoramento de ambientes e resolução de incidentes técnicos. Foco em arquitetura escalável, automação de processos, boas práticas de código e disponibilidade de sistemas.
            """)

            st.markdown('<div class="section-title">Experiência Profissional</div>', unsafe_allow_html=True)
            st.markdown("""
            * **Estagiário de TI — FIAP** *(Novembro de 2025 – Presente)*
              * Atuação em suporte técnico N1 e monitoramento da infraestrutura de TI e redes dos laboratórios acadêmicos.
              * Diagnóstico e resolução de incidentes de hardware e software, garantindo alta disponibilidade dos equipamentos e continuidade operacional.
              * Atendimento a chamados corporativos com foco na redução do tempo de resposta e padronização de rotinas de suporte.
            """)
            
            st.markdown('<div class="section-title">Premiações & Reconhecimentos</div>', unsafe_allow_html=True)
            st.markdown("""
            * **1º Lugar no FIAP Hackathon 2026 — Projeto Auto-Lab:**
              * Desenvolvimento de solução integrada para inventário automatizado de ativos e gerenciamento remoto de infraestrutura de laboratórios de TI.
              * Implementação de rotinas para acionamento remoto de máquinas via rede (Wake-on-LAN), diagnósticos automatizados e consolidação de relatórios técnicos.
            """)
            
            st.markdown('<div class="section-title">Projetos em Destaque</div>', unsafe_allow_html=True)
            st.markdown("""
            * **Plataforma de Inteligência Antifraude & Modelagem de Risco:**
              * Dashboard analítico desenvolvido em Python/Streamlit com tratamento de bases transacionais desbalanceadas, inferência estatística (Intervalos de Confiança a 95%) e simulador de limiar de decisão (*Threshold Tuning*) via Scikit-Learn.
            * **Aplicações Web & SPAs:**
              * Desenvolvimento de interfaces modulares utilizando React.js, Tailwind CSS e integração de APIs RESTful.
            * **Certificações:** *Design Thinking Process* — FIAP.
            """)

        st.markdown('</div>', unsafe_allow_html=True)

    def render(self):
        self.aplicar_estilo()
        self.renderizar_cabecalho()
        self.renderizar_conteudo()

if __name__ == "__main__":
    app = CurriculoPage()
    app.render()