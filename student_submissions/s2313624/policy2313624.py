# get_action với generate_cutting_pattern là quan trọng nhất
# mấy hàm còn lại mang tính chất phụ trợ, không cần đào sâu
# trong báo cáo

from policy import Policy
import numpy as np

class Policy2313624(Policy):
    def __init__(self):
        super().__init__()

    # trả về kích thước của stock
    def _get_stock_size_(self, stock):
        return super()._get_stock_size_(stock)

    # kiểm tra xem vị trí đó có thể bị thay thế
    def _can_place_(self, stock, position, prod_size):
        return super()._can_place_(stock, position, prod_size)

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        # mảng này chứa các mẫu cắt
        patterns = []

        # khởi tạo chi phí tối thiểu
        min_cost = float("inf")
        best_action = {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        # vòng lặp giải thuật column generation
        while True:
            # tạo mẫu cắt mới dựa 
            new_pattern, cost = self._generate_cutting_pattern(list_prods, stocks)

            # nếu không tìm được mẫu cắt mới hoặc chi phí tệ hơn thì out
            if new_pattern is None or cost >= min_cost:
                break

            # thêm vào các mẫu cắt
            patterns.append(new_pattern)

            # cập nhật chi phi min và action tốt nhất
            min_cost = cost
            best_action = new_pattern["action"]

        return best_action

    def _generate_cutting_pattern(self, list_prods, stocks):
        # tìm một mẫu cắt mới bằng cách duyệt qua các sản phẩm và stocks
        # trả về mẫu cắt mới và chi phí
        min_trim_loss = float("inf")
        best_pattern = None

        # sắp xếp theo kích thước giảm dần
        sorted_prods = sorted(list_prods, key=lambda x: max(x["size"]), reverse=True)

        # duyệt qua từng sản phẩm
        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # duyệt qua từng stock
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # kiểm tra xem sản phẩm fit vào stock không
                    if stock_w < prod_w and stock_h < prod_h:
                        continue

                    # duyệt qua tất cả các vị trí có thể đặt sản phẩm
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break

                    # nếu vị trí hợp lệ thì tính trim_loss
                    if pos_x is not None and pos_y is not None:
                        # tạo bản sao của stock để cập nhật
                        stock_copy = stock.copy()
                        stock_copy[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] = -2  # -2 là ô đã sử dụng

                        # tính trim loss
                        trim_loss = self._calculate_trim_loss(stock_copy)

                        # nếu trim loss thấp hơn, cập nhật mẫu
                        if trim_loss < min_trim_loss:
                            min_trim_loss = trim_loss
                            best_pattern = {
                                "action": {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (pos_x, pos_y),
                                },
                                "trim_loss": trim_loss
                            }

        if best_pattern is not None:
            return best_pattern, min_trim_loss
        else:
            return None, float("inf")


    # tính toán trim loss
    def _calculate_trim_loss(self, stock):
        trim_loss = np.sum(stock == -1)  # ô trống được đánh dấu là -1
        return trim_loss
