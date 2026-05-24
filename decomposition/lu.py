import numpy as np
import scipy.linalg
import streamlit as st

def run_lu_workflow(matrix: np.ndarray):

    st.markdown("---")
    st.subheader("🤖 LU Decomposition Results")
    

    if matrix.shape[0] != matrix.shape[1]:
        st.error("❌ Validation Error: LU Decomposition can only be performed on a square matrix ($N \times N$).")
        return
        
    try:

        P, L, U = scipy.linalg.lu(matrix)

        reconstructed = P @ L @ U
        rec_error = float(np.linalg.norm(matrix - reconstructed))
        
        st.success("🎉 LU Decomposition completed successfully!")
        

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Permutation Matrix ($P$)**")
            st.dataframe(P)
        with col2:
            st.write("**Lower Triangular ($L$)**")
            st.dataframe(L)
        with col3:
            st.write("**Upper Triangular ($U$)**")
            st.dataframe(U)
            

        st.markdown("### 🔍 Reconstruction Verification")
        st.metric(label="Reconstruction Error (Frobenius Norm)", value=f"{rec_error:.2e}")
        
        if np.allclose(matrix, reconstructed, atol=1e-9):
            st.info("✅ **Verification Pass:** $P \times L \times U$ perfectly reconstructs the original matrix $A$.")
        else:
            st.warning("⚠️ **Verification Warning:** High reconstruction error detected. Check matrix numerical stability.")
            
    except Exception as e:
        st.error(f"❌ An unexpected numerical error occurred during computation: {str(e)}")
