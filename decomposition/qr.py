import numpy as np
import streamlit as st

def run_qr_workflow(matrix: np.ndarray):

    st.markdown("---")
    st.subheader("📐 QR Decomposition Results")
    
    try:

        Q, R = np.linalg.qr(matrix)
        

        reconstructed = Q @ R
        rec_error = float(np.linalg.norm(matrix - reconstructed))
        
        st.success("🎉 QR Decomposition completed successfully!")
        

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Orthogonal Matrix ($Q$)**")
            st.dataframe(Q)
        with col2:
            st.write("**Upper Triangular Matrix ($R$)**")
            st.dataframe(R)
            

        st.markdown("### 🔍 Reconstruction Verification")
        st.metric(label="Reconstruction Error (Frobenius Norm)", value=f"{rec_error:.2e}")
        
        if np.allclose(matrix, reconstructed, atol=1e-9):
            st.info("✅ **Verification Pass:** $Q \times R$ perfectly reconstructs the original matrix $A$.")
        else:
            st.warning("⚠️ **Verification Warning:** High reconstruction error detected.")
            
    except Exception as e:
        st.error(f"❌ An unexpected numerical error occurred during computation: {str(e)}")
