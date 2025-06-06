| Name | Description |
|--------------------|-------------------|
| Сумма | Total order amount for delivery |
| Материал | Material ID |
| Поставщик | Supplier ID |
| Категорийный менеджер | Category manager ID |
| Операционный менеджер | Operational manager ID |
| Завод | Plant ID |
| Закупочная организация | Purchasing organization ID |
| Балансовая единица | Balancing unit ID |
| Группа закупок | Purchasing group ID |
| Группа материалов | Material group ID |
| Вариант поставки | Delivery option ID |
| НРП | Number of times the order was considered urgent. In rare cases, it may be more than once |
| Длительность | Difference in days between the creation of the order and the planned start of delivery |
| Месяц 1 | Month of order creation |
| Месяц 2 | Month when delivery started |
| День недели 2 | Day of the week delivery started |
| Изменение даты поставки 30/15/7 | Number of times the delivery date changed within 30/15/7 days from the order creation date |
| Количество | Quantity of goods |
| Согласование заказа 1 | Frequency of order being at stage 1 |
| Согласование заказа 2 | Frequency of order being at stage 2 |
| Согласование заказа 3 | Frequency of order being at stage 3 |
| Изменение позиции заказа на закупку: дата поставки | Frequency of the event “Change of order position: delivery date” |
| Отмена полного деблокирования заказа на закупку | Frequency of the event “Full cancellation of purchase order unblocking” |
| Изменение позиции заказа на закупку: изменение даты поставки на бумаге | Frequency of the event “Change of order position: change of delivery date on paper.” Supplier changes on documents |
| Дней между 0-1 | Number of days between first events: “Order position creation” and “Order position creation” |
| Дней между 1-2 | Number of days between events: “Order position creation” and “Order approval stage 1” |
| Дней между 2-3 | Number of days between events: “Order approval stage 1” and “Order approval stage 2” |
| Дней между 3-4 | Number of days between events: “Order approval stage 2” and “Order approval stage 3” |
| Дней между 4-5 | Number of days between events: “Order approval stage 3” and “Full unblocking of the order” |
| Дней между 5-6 | Number of days between events: “Full unblocking of the order” and “Change of order position: change of planned delivery date” |
| Дней между 6-7 | Number of days between events: “Change of order position: change of planned delivery date” and “Change of order position: change of delivery date on paper” |
| Дней между 7-8 | Number of days between events: “Change of order position: change of delivery date on paper” and “Change of purchase request: assignment of delivery source” |
| Количество обработчиков 7 | Number of unique users who worked with the order within 7 days of its creation |
| Количество обработчиков 15 | Number of unique users who worked with the order within 15 days of its creation |
| Количество обработчиков 30 | Number of unique users who worked with the order within 30 days of its creation |
| ЕИ | Unit of measurement ID |
| До поставки | Minimum number of days between receipt and planned delivery/duration. If this value is less than the value in the “Длительность” column, the delivery occurred earlier than planned |
| Количество циклов согласования | Number of approval cycles through all stages |
| Количество изменений после согласований | Number of changes made to the order after approvals |
| Месяц 3 | Month when the material first appeared in the order in legacy systems. Can be ignored during analysis |
| Количество позиций | Number of order positions. Recommended not to be considered during analysis |
