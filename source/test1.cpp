
#include "aogmaneo/array.h"
#include "aogmaneo/helpers.h"
#include "aogmaneo/hierarchy.h"
#include <cstring>
int main() {
  using namespace aon;
  Array<Hierarchy::IO_Desc> ios(2);
  ios[0] = Hierarchy::IO_Desc(Int3(1, 4, 32), prediction);
  ios[1] = Hierarchy::IO_Desc(Int3(1, 1, 2), action);

  Array<Hierarchy::Layer_Desc> descs(2);
  descs[0] = descs[1] = Hierarchy::Layer_Desc();
  Hierarchy hier(ios, descs);

  auto predictions = hier.get_prediction_cis(1);
  Array<int> obs_a(4);
  obs_a[0] = 20;
  obs_a[1] = 21;
  obs_a[2] = 22;
  obs_a[3] = 23;
  Int_Buffer_View input_cis[2] = {
      obs_a,
      predictions,
  };
  const Array<Int_Buffer_View> obs_aa(2);
  std::memcpy(obs_aa.p, input_cis, 2);
  hier.step(obs_aa);

  return 0;
}
