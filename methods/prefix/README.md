# prefix

保留合法前缀与从起点重启的独立消融。

入口：`python -m methods.prefix <command> -- <原入口参数>`。只查看配置和模块归属：`python -m methods.prefix --describe`。

可用命令：`run`。详细映射见 [manifest.json](manifest.json)。配置引用仓库原文件，不复制训练参数或改变默认值。底层实现保留原路径，防止历史导入及冻结证据失效。
